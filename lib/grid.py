import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from . import dvgo, dcvgo, dmpigo

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)


def create_grid(type, **kwargs):
    if type == 'DenseGrid':
        return DenseGrid(**kwargs)
    elif type == 'TensoRFGrid':
        return TensoRFGrid(**kwargs)
    else:
        raise NotImplementedError


''' Dense 3D grid
'''
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

        # defined to save the indices for updating uncertainty_grid
        self.indices_to_update = None
    
    def Normalize(self, xyz):
        return (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1 # * 2 - 1 is because grid_sample use range [-1,1], flip((-1,)) may be because grid_sample takes (z,y,z) as input?
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_cuda.total_variation_add_grad(
            self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'
    
    @torch.no_grad()
    def reset(self, new_world_size, datatype):
        self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))

        if datatype == 'unbounded':
            # unbounded scenes involves ray parameterization for the unbounded background and thus need more complex initialization!
            # firstly, init high value in the uncertainty field for the non-contracted space (foreground space, [-1,1]), involving points O,E,S in the paper
            normalized_fbbox_lower = self.Normalize(torch.tensor([-1,-1,-1]))
            normalized_fbbox_upper = self.Normalize(torch.tensor([1,1,1]))
            fbbox_indices_lower = torch.round(normalized_fbbox_lower * new_world_size).long()
            fbbox_indices_upper = torch.round(normalized_fbbox_upper * new_world_size).long()
            self.grid.data[:,:,fbbox_indices_lower[0]:fbbox_indices_upper[0],fbbox_indices_lower[1]:fbbox_indices_upper[1],fbbox_indices_lower[2]:fbbox_indices_upper[2]] += 1
            # Secondly, init high uncertainty value for only boundary of the contracted space (background space), involving points P in the paper  
            pad = 3
            self.grid.data[:,:,:pad] += 1
            self.grid.data[:,:,-pad:] += 1
            self.grid.data[:,:,:,:pad] += 1
            self.grid.data[:,:,:,-pad:] += 1
            self.grid.data[:,:,:,:,:pad] += 1
            self.grid.data[:,:,:,:,-pad:] += 1

        if datatype == 'bounded':
            self.grid.data += 1

        print('reset uncertainty grid:',(self.grid.data==0).sum()/torch.prod(new_world_size))

    # Used for update the uncertainty grid with the sampled points
    @torch.no_grad()
    def set_zero_at(self, xyz, new_world_size):
        # Scale xyz from range (xyz_min, xyz_max) to (0,1)
        normalized_xyz = self.Normalize(xyz)

        # Convert normalized values to grid indices
        base_indices = torch.unique(torch.round(normalized_xyz * (new_world_size - 1)).long(), dim=0)

        if self.indices_to_update == None:
            self.indices_to_update = base_indices
        else:
            self.indices_to_update = torch.unique(torch.cat([self.indices_to_update, base_indices],0), dim=0)
    
    @torch.no_grad()
    def update_indices(self, new_world_size):
        '''
        Given a ray point, set the nearest vertex to be of zero uncertainty 
        '''
        if self.indices_to_update is None:
            return
        # # Broadcast to get all the indices for the 2x2x2 neighbors for each point
        # offsets = torch.tensor([
        #     [0, 0, 0],
        #     [0, 0, 1],
        #     [0, 1, 0],
        #     [0, 1, 1],
        #     [1, 0, 0],
        #     [1, 0, 1],
        #     [1, 1, 0],
        #     [1, 1, 1]
        # ])
        # all_indices = self.indices_to_update[:, None] + offsets
        # unique_indices = torch.unique(all_indices.reshape(-1,3), dim=0)

        # Ensure the indices are within the grid boundaries
        unique_indices = torch.clamp(self.indices_to_update, max=new_world_size-1)
        # print('unique_indices updated: ',unique_indices.shape[0]/torch.prod(new_world_size))

        # Update the values at these indices to 0
        CHUNK = 10000
        for i in range(0, unique_indices.shape[0], CHUNK):
            self.grid.data[:, :, unique_indices[i:i+CHUNK, 0], unique_indices[i:i+CHUNK, 1], unique_indices[i:i+CHUNK, 2]] = 0  
        # print('uncertainty_grid updated: ',(self.grid.data==0).sum()/torch.prod(new_world_size))

        # Ensure the indices on the grid boundaries are zero
        self.grid.data[:,:,:1] = 0
        self.grid.data[:,:,-1:] = 0
        self.grid.data[:,:,:,:1] = 0
        self.grid.data[:,:,:,-1:] = 0
        self.grid.data[:,:,:,:,:1] = 0
        self.grid.data[:,:,:,:,-1:] = 0

        self.indices_to_update = None

    @torch.no_grad()
    def update_indices_unseen_foreground(self, new_world_size):
        if self.indices_to_update is None:
            return
        
        unique_indices = torch.clamp(self.indices_to_update,  max=new_world_size-1)

        CHUNK = 10000
        for i in range(0, unique_indices.shape[0], CHUNK):
            self.grid.data[:, :, unique_indices[i:i+CHUNK, 0], unique_indices[i:i+CHUNK, 1], unique_indices[i:i+CHUNK, 2]] += 1 
        
        
        normalized_fbbox_lower = self.Normalize(torch.tensor([-1,-1,-1]))
        normalized_fbbox_upper = self.Normalize(torch.tensor([1,1,1]))
        fbbox_indices_lower = torch.round(normalized_fbbox_lower * new_world_size).long()
        fbbox_indices_upper = torch.round(normalized_fbbox_upper * new_world_size).long()
        
        temp = self.grid.data[:,:,fbbox_indices_lower[0]:fbbox_indices_upper[0],fbbox_indices_lower[1]:fbbox_indices_upper[1],fbbox_indices_lower[2]:fbbox_indices_upper[2]]
        temp[temp==1] = 0

        self.indices_to_update = None

        return 
    
    @torch.no_grad()
    def maskout(self, xyz_fine_min, xyz_fine_max):

        shape = torch.tensor([*self.grid.data.shape[2:]])-1
        
        def normalize(xyz): return ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * shape).int()
        
        xyz_fine_min_norm = normalize(xyz_fine_min)
        xyz_fine_max_norm = normalize(xyz_fine_max)

        self.grid.data[:,:,:xyz_fine_min_norm[0]] = 0 
        self.grid.data[:,:,:,:xyz_fine_min_norm[1],:] = 0 
        self.grid.data[:,:,:,:,:xyz_fine_min_norm[2]] = 0 
        self.grid.data[:,:,xyz_fine_max_norm[0]:] = 0 
        self.grid.data[:,:,:,xyz_fine_max_norm[1]:] = 0
        self.grid.data[:,:,:,:,xyz_fine_max_norm[2]:] = 0        

''' Mask grid
It supports query for the known free space and unknown space.
'''
class MaskGrid(nn.Module):
    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(st['model_state_dict']['density.grid'], kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(-F.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])

            # merge high uncertain area in mask_cache
            # print('use high uncertain area in mask_cache!')
            # uncertainty = F.max_pool3d(st['model_state_dict']['uncertainty_grid.grid'], kernel_size=3, padding=1, stride=1)
            # mask |= (uncertainty>0).squeeze(0).squeeze(0)

        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'