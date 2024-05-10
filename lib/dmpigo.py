import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from torch_scatter import scatter_add, segment_coo

from . import grid
from .dvgo import *


'''Model'''
class DirectMPIGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, mpi_depth=0,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=0,
                 **kwargs):
        super(DirectMPIGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres

        # determine init grid resolution
        self._set_grid_resolution(num_voxels, mpi_depth)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
                density_type, channels=1, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.density_config)

        # init density bias so that the initial contribution (the alpha values)
        # of each query points on a ray is equal
        self.act_shift = grid.DenseGrid(
                channels=1, world_size=[1,1,mpi_depth],
                xyz_min=xyz_min, xyz_max=xyz_max)
        self.act_shift.grid.requires_grad = False
        with torch.no_grad():
            g = np.full([mpi_depth], 1./mpi_depth - 1e-6)
            p = [1-g[0]]
            for i in range(1, len(g)):
                p.append((1-g[:i+1].sum())/(1-g[:i].sum()))
            for i in range(len(p)):
                self.act_shift.grid[..., i].fill_(np.log(p[i] ** (-1/self.voxel_size_ratio) - 1))

        # init color representation
        # feature voxel grid + shallow MLP  (fine stage)
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2) + self.k0_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)

        print('dmpigo: densitye grid', self.density)
        print('dmpigo: feature grid', self.k0)
        print('dmpigo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels, mpi_depth):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256. / mpi_depth
        print('dmpigo: world_size      ', self.world_size)
        print('dmpigo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'mpi_depth': self.mpi_depth,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, mpi_depth):
        print('dmpigo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels, mpi_depth)
        print('dmpigo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            dens = self.density.get_dense_grid() + self.act_shift.grid
            self_alpha = F.max_pool3d(self.activate_density(dens), kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                    path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dmpigo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        print('dmpigo: update mask_cache lt_nviews start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, ray_id, step_id, N_samples = self.sample_ray(
                        rays_o=rays_o.to(device), rays_d=rays_d.to(device), **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += (ones.grid.grad > 1)
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0,0]
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')
        torch.cuda.empty_cache()
        eps_time = time.time() - eps_time
        print(f'dmpigo: update mask_cache lt_nviews finish (eps time:', eps_time, 'sec)')

    def density_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.density.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.k0.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), 0, interval).reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        assert near==0 and far==1
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N_samples = int((self.mpi_depth-1)/stepsize) + 1
        ray_pts, mask_outbbox = render_utils_cuda.sample_ndc_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, N_samples)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        if mask_inbbox.all():
            ray_id, step_id = create_full_step_id(mask_inbbox.shape)
        else:
            ray_id = torch.arange(mask_inbbox.shape[0]).view(-1,1).expand_as(mask_inbbox)[mask_inbbox]
            step_id = torch.arange(mask_inbbox.shape[1]).view(1,-1).expand_as(mask_inbbox)[mask_inbbox]
        return ray_pts, ray_id, step_id, N_samples

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id, N_samples = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for alpha w/ post-activation
        density = self.density(ray_pts) + self.act_shift(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            weights = weights[mask]

        # query for color
        vox_emb = self.k0(ray_pts)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(vox_emb)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb[ray_id]
            rgb_feat = torch.cat([vox_emb, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            rgb = torch.sigmoid(rgb_logit)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and global_step is not None:
            rgb_marched += (alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched))
        else:
            rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        s = (step_id+0.5) / N_samples
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
            'n_max': N_samples,
            's': s,
            'ray_pts':ray_pts,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * s),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'render_depth': depth})

        return ret_dict

class DirectMPIGO_flow0(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, mpi_depth=0,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=0,
                 density_std_init=1,
                 K_samples=12,
                 **kwargs):
        super(DirectMPIGO_flow, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres

        self.density_std_init = density_std_init

        self.glob_lambda_den = 0.
        self.glob_lambda_rgb = 0.

        # determine init grid resolution
        self._set_grid_resolution(num_voxels, mpi_depth)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
                density_type, channels=1, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.density_config)
        
        self.density_std = grid.create_grid(
                density_type, channels=1, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.density_config)
        with torch.no_grad():
            for i in range(mpi_depth):
                self.density_std.grid[..., i].fill_(np.log(np.exp(density_std_init * i / mpi_depth+0.1) - 1))

        # init density bias so that the initial contribution (the alpha values)
        # of each query points on a ray is equal
        self.act_shift = grid.DenseGrid(
                channels=1, world_size=[1,1,mpi_depth],
                xyz_min=xyz_min, xyz_max=xyz_max)
        self.act_shift.grid.requires_grad = False
        with torch.no_grad():
            g = np.full([mpi_depth], 1./mpi_depth - 1e-6)
            p = [1-g[0]]
            for i in range(1, len(g)):
                p.append((1-g[:i+1].sum())/(1-g[:i].sum()))
            for i in range(len(p)):
                self.act_shift.grid[..., i].fill_(np.log(p[i] ** (-1/self.voxel_size_ratio) - 1) - 1)

        # init color representation
        # feature voxel grid + shallow MLP  (fine stage)
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2) + self.k0_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 6),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)

        print('dmpigo: densitye grid', self.density)
        print('dmpigo: feature grid', self.k0)
        print('dmpigo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        
        # init for our add flow
        self.K_samples = K_samples

        self.global_flow_den = torch.nn.Parameter(torch.zeros(2))
        self.global_flow_rgb = torch.nn.Parameter(torch.zeros(6))

        self.sample_den = torch.empty([K_samples]).normal_()
        self.sample_rgb = torch.empty([K_samples,3]).normal_()

    def _set_grid_resolution(self, num_voxels, mpi_depth):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256. / mpi_depth
        print('dmpigo: world_size      ', self.world_size)
        print('dmpigo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'mpi_depth': self.mpi_depth,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            'density_std_init': self.density_std_init,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, mpi_depth):
        print('dmpigo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels, mpi_depth)
        print('dmpigo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)
        self.density_std.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)

            cache_grid_density = self.density(self_grid_xyz) # (*world_size)
            cache_grid_density_std = self.density_std(self_grid_xyz) # (*world_size)
            print('cache_grid_density',cache_grid_density.max(),cache_grid_density.min())
            print('cache_grid_density_std',cache_grid_density_std.max(),cache_grid_density_std.min())
            den_glob = self.global_flow_den[0:1][None,None,:].expand_as(cache_grid_density)
            den_joint = cache_grid_density + den_glob * self.glob_lambda_den

            # den_std = F.softplus(cache_grid_density_std) + 1e-03
            # den_glob_std = self.global_flow_den[1:2][None,None,:].expand_as(den_std)
            # den_glob_std = F.softplus(den_glob_std) + 1e-03
            # den_joint_std = (den_std**2 + (den_glob_std * self.glob_lambda_rgb)**2).sqrt()
            # eps_den = torch.Tensor([1.])
            # den_joint = eps_den.mul(den_joint_std).add_(den_joint)
            
            self_alpha = F.max_pool3d(self.activate_density(den_joint + self.act_shift.grid), kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                    path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dmpigo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        
        cache_grid_density = self.density(cache_grid_xyz) # (*world_size)
        cache_grid_density_std = self.density_std(cache_grid_xyz) # (*world_size)
        print('cache_grid_density',cache_grid_density.max(),cache_grid_density.min())
        print('cache_grid_density_std',cache_grid_density_std.max(),cache_grid_density_std.min())
        den_glob = self.global_flow_den[0:1][None,None,:].expand_as(cache_grid_density)
        den_joint = cache_grid_density + den_glob * self.glob_lambda_den

        # den_std = F.softplus(cache_grid_density_std) + 1e-03
        # den_glob_std = self.global_flow_den[1:2][None,None,:].expand_as(den_std)
        # den_glob_std = F.softplus(den_glob_std) + 1e-03
        # den_joint_std = (den_std**2 + (den_glob_std * self.glob_lambda_rgb)**2).sqrt()
        # eps_den = torch.Tensor([1.])
        # den_joint = eps_den.mul(den_joint_std).add_(den_joint)

        cache_grid_alpha = self.activate_density(den_joint + self.act_shift.grid)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        print('dmpigo: update mask_cache lt_nviews start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, ray_id, step_id, N_samples = self.sample_ray(
                        rays_o=rays_o.to(device), rays_d=rays_d.to(device), **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += (ones.grid.grad > 1)
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0,0]
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')
        torch.cuda.empty_cache()
        eps_time = time.time() - eps_time
        print(f'dmpigo: update mask_cache lt_nviews finish (eps time:', eps_time, 'sec)')

    def density_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.density.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def params_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.density_std.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.k0.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), 0, interval).reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        assert near==0 and far==1
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N_samples = int((self.mpi_depth-1)/stepsize) + 1

        ray_pts, mask_outbbox = render_utils_cuda.sample_ndc_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, N_samples)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        if mask_inbbox.all():
            ray_id, step_id = create_full_step_id(mask_inbbox.shape)
        else:
            ray_id = torch.arange(mask_inbbox.shape[0]).view(-1,1).expand_as(mask_inbbox)[mask_inbbox]
            step_id = torch.arange(mask_inbbox.shape[1]).view(1,-1).expand_as(mask_inbbox)[mask_inbbox]
        return ray_pts, ray_id, step_id, N_samples

    def forward(self, rays_o, rays_d, viewdirs, global_step=1000, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        self.is_test = render_kwargs['is_test']

        # sample points on rays
        ray_pts, ray_id, step_id, N_samples = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for alpha w/ post-activation
        density_mean = self.density(ray_pts)
        density_std = self.density_std(ray_pts)
        density_std = F.softplus(density_std) + 1e-3

        # den_glob, den_glob_std = self.global_flow_den[:1], self.global_flow_den[1:]
        # den_glob_std = F.softplus(den_glob_std) + 1e-3

        # # sample density, sum of coordinate-based distribution and global distribution, both are gaussian
        # den_mean = density_mean + den_glob * self.glob_lambda_den
        # den_std = (density_std**2 + (den_glob_std * self.glob_lambda_den)**2).sqrt()

        # loss_entropy = - den_std.log()
        
        if not self.is_test:
            eps_den = torch.empty([1,self.K_samples]).normal_()
        else:
            eps_den = self.sample_den[None]

        if global_step > 900:
            raw_density_k = eps_den.mul(density_std[:,None]).add_(density_mean[:,None]) # (BN,K,1)
        else:
            BN = density_mean.shape[0]
            raw_density_k = density_mean[:,None].expand([BN, self.K_samples])
        density_k = F.softplus(raw_density_k + self.act_shift(ray_pts)[:,None])
        # alpha_k = self.activate_density(raw_density_k + self.act_shift(ray_pts)[:,None], interval)
        alpha_k = self.activate_density(raw_density_k + self.act_shift(ray_pts)[:,None], interval)
        alpha = alpha_k.mean(-1)

        if self.fast_color_thres > 0:
            alpha_std = alpha_k.std(-1)
            mask = (alpha > self.fast_color_thres)
            alpha_k = alpha_k[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            alpha_std = alpha_std[mask]
            density_k = density_k[mask]
            # loss_entropy = loss_entropy[mask]

        # compute accumulated transmittance
        shape = alpha_k.shape
        alpha_std_sum = segment_coo(src=(alpha_std),index=ray_id,out=torch.zeros([N]),reduce='sum')
        weights_k, alphainv_last_k = Alphas2Weights_k.apply(alpha_k, ray_id, N, alpha_std.detach(), alpha_std_sum.detach())
        weights_k = weights_k.reshape(shape)
        weights = weights_k.mean(-1)
        alphainv_last_k = alphainv_last_k.reshape([N,self.K_samples])
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights_std = weights_k.std(-1)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            weights = weights[mask]
            weights_k = weights_k[mask]
            density_k = density_k[mask]
            # loss_entropy = loss_entropy[mask]

        # query for color
        vox_emb = self.k0(ray_pts)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(vox_emb)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb[ray_id]
            rgb_feat = torch.cat([vox_emb, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)

            rgb_mean, rgb_std = rgb_logit[:,:3], rgb_logit[:,3:]
            rgb_std = F.softplus(rgb_std) + 1e-3
            # rgb_std = torch.zeros_like(rgb_std)

            # rgb_glob, rgb_glob_std = self.global_flow_rgb[:3], self.global_flow_rgb[3:6]
            # rgb_glob_std = F.softplus(rgb_glob_std) + 1e-3
            # # sample rgb, sum of coordinate-based distribution and global distribution, both are gaussian
            # rgb_mean = rgb_mean + rgb_glob * self.glob_lambda_rgb
            # rgb_std = (rgb_std**2 + (rgb_glob_std * self.glob_lambda_rgb)**2).sqrt()
            # loss_entropy += - rgb_std.log().mean(-1)

            if not render_kwargs['is_test']:
                eps_rgb = torch.empty([1,self.K_samples, 3]).normal_()
            else:
                eps_rgb = self.sample_rgb[None]

            if global_step > 900:
                raw_rgb_k = eps_rgb.mul(rgb_std[:,None]).add_(rgb_mean[:,None]) # (BN,K,3)
            else:
                BN = rgb_mean.shape[0]
                raw_rgb_k = rgb_mean[:,None].expand([BN, self.K_samples, 3])
            rgb_k = torch.sigmoid(raw_rgb_k)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights_k[...,None] * rgb_k),
                index=ray_id,
                out=torch.zeros([N, self.K_samples, 3]),
                reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and global_step is not None:
            rgb_marched += (alphainv_last_k.unsqueeze(-1) * torch.rand_like(rgb_marched))
        else:
            rgb_marched += (alphainv_last_k.unsqueeze(-1) * render_kwargs['bg'])
        s = (step_id+0.5) / N_samples
        ret_dict.update({
            'alphainv_last': alphainv_last_k.mean(-1),
            'alphainv_last_k': alphainv_last_k,
            'weights': weights,
            'weights_k': weights_k,
            'rgb_marched': rgb_marched,
            'density_k': density_k,
            'raw_rgb_k': rgb_k,
            'ray_id': ray_id,
            'n_max': N_samples,
            's': s,
            'ray_pts': ray_pts,
            # 'loss_entropy': loss_entropy,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth_k = segment_coo(
                        src=(weights_k * s.unsqueeze(-1)),
                        index=ray_id,
                        out=torch.zeros([N, self.K_samples]),
                        reduce='sum')
                # acc_map = segment_coo(
                #         src=weights_k,
                #         index=ray_id,
                #         out=torch.zeros([N, self.K_samples]),
                #         reduce='sum')
                # disp_k = 1./torch.max(1e-10 * torch.ones_like(depth_k), depth_k / acc_map)
                # print('depth_k',depth_k.max(),depth_k.min())
                # print('acc_map',acc_map.max(),acc_map.min())
                # print('disp_k',disp_k.max(),disp_k.min())
            ret_dict.update({'render_depth': depth_k})
        
        # np.save('./logs/llff/trex/trex_snerf/ray_pts1.npy', ray_pts.detach().cpu().numpy())
        # print('ray_pts:',ray_pts.shape)
        # exit()

        return ret_dict

class DirectMPIGO_flow1(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, mpi_depth=0,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=0,
                 K_samples=12,
                 num_flows=[4,4],
                 density_std_init=1,
                 glob_lambda_den=0.01,
                 **kwargs):
        super(DirectMPIGO_flow, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        denfea_dim = num_flows[0] * 4
        rgbfea_dim = num_flows[1] * (3*3+3+3+3) + 6
        self.num_flows = num_flows
        self.density_std_init = density_std_init
        self.glob_lambda_den = glob_lambda_den
        self.glob_lambda_rgb = glob_lambda_den

        print('K_samples',K_samples)
        print('density_std_init',density_std_init)
        print('num_flows',num_flows)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels, mpi_depth)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        
        # init params voxel grid
        self.density_params = grid.create_grid(
            density_type, channels=denfea_dim, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            config=self.density_config)
        with torch.no_grad():
            self.density_params.grid.data.normal_(0, 0.05)

        # init density bias so that the initial contribution (the alpha values)
        # of each query points on a ray is equal
        self.act_shift = grid.DenseGrid(
                channels=1, world_size=[1,1,mpi_depth],
                xyz_min=xyz_min, xyz_max=xyz_max)
        self.act_shift.grid.requires_grad = False
        with torch.no_grad():
            g = np.full([mpi_depth], 1./mpi_depth - 1e-6)
            p = [1-g[0]]
            for i in range(1, len(g)):
                p.append((1-g[:i+1].sum())/(1-g[:i].sum()))
            for i in range(len(p)):
                self.act_shift.grid[..., i].fill_(np.log(p[i] ** (-1/self.voxel_size_ratio) - 1) - 1)

        # init color representation
        # feature voxel grid + shallow MLP  (fine stage)
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2) + self.k0_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, rgbfea_dim),
            )
            # nn.init.constant_(self.rgbnet[-1].bias, 0)

        print('dmpigo: feature grid', self.k0)
        print('dmpigo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        
         # init for our add flow
        self.K_samples = K_samples

        self.global_flow_den = torch.nn.Parameter(torch.zeros(2))
        self.global_flow_rgb = torch.nn.Parameter(torch.zeros(6))

        self.sample_size = self.K_samples
        self.sample_den = torch.empty([self.sample_size,1]).normal_()
        self.sample_rgb = torch.empty([self.sample_size,3]).normal_()
    
        self.flows_den = TriangularSylvesterNeRF_Voxel(num_flows[0],'density')
        self.flows_rgb = TriangularSylvesterNeRF_Voxel(num_flows[1],'rgb')

    def _set_grid_resolution(self, num_voxels, mpi_depth):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256. / mpi_depth
        print('dmpigo: world_size      ', self.world_size)
        print('dmpigo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'mpi_depth': self.mpi_depth,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            'K_samples': self.K_samples,
            'num_flows': self.num_flows,
            'density_std_init': self.density_std_init,
            'glob_lambda_den': self.glob_lambda_den,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, mpi_depth):
        print('dmpigo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels, mpi_depth)
        print('dmpigo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density_params.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            
            cache_grid_density_params = self.density_params.get_dense_grid().permute(0,2,3,4,1)[0] # (*world_size,channels)
            shape = cache_grid_density_params.shape[:3]
            den_mean = self.global_flow_den[:1,None].expand([*cache_grid_density_params.shape[:3], 1]).reshape([-1,1])
            params = cache_grid_density_params.view([-1,cache_grid_density_params.shape[-1]])
            z_den, _ = self.flows_den(den_mean, params)
            cache_grid_alpha = self.activate_density(z_den.reshape(shape))[None,None]
            self_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)
            
            ori_p = self.mask_cache.mask.float().mean().item()
            print(f'dcvgo: mask_cache after scale {ori_p:.4f}')

            if ori_p < 0.75 and self.global_flow_den.requires_grad:
                print('set grad of global_flow_den to be False')
                self.global_flow_den.requires_grad = False
                # self.global_flow_rgb.requires_grad = False

        print('dmpigo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)

        cache_grid_density_params = self.density_params(cache_grid_xyz) # (*world_size,channels)
        shape = cache_grid_density_params.shape[:3]
        den_mean = self.global_flow_den[:1,None].expand([*cache_grid_density_params.shape[:3], 1]).reshape([-1,1])
        params = cache_grid_density_params.view([-1,cache_grid_density_params.shape[-1]])
        z_den, _ = self.flows_den(den_mean, params)
        cache_grid_alpha = self.activate_density(z_den.reshape(shape))[None,None]
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dcvgo: update mask_cache {ori_p:.4f} => {new_p:.4f}')

        if new_p < 0.75 and self.global_flow_den.requires_grad:
            print('set grad of global_flow_den to be False')
            self.global_flow_den.requires_grad = False
            # self.global_flow_rgb.requires_grad = False

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        print('dmpigo: update mask_cache lt_nviews start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, ray_id, step_id, N_samples = self.sample_ray(
                        rays_o=rays_o.to(device), rays_d=rays_d.to(device), **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += (ones.grid.grad > 1)
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0,0]
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')
        torch.cuda.empty_cache()
        eps_time = time.time() - eps_time
        print(f'dmpigo: update mask_cache lt_nviews finish (eps time:', eps_time, 'sec)')

    def density_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        # self.density.total_variation_add_grad(wxy, wxy, wz, dense_mode)
    
    def params_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        # self.density_std.total_variation_add_grad(wxy, wxy, wz, dense_mode)
        self.density_params.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.k0.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), 0, interval).reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        assert near==0 and far==1
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N_samples = int((self.mpi_depth-1)/stepsize) + 1
        ray_pts, mask_outbbox = render_utils_cuda.sample_ndc_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, N_samples)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        if mask_inbbox.all():
            ray_id, step_id = create_full_step_id(mask_inbbox.shape)
        else:
            ray_id = torch.arange(mask_inbbox.shape[0]).view(-1,1).expand_as(mask_inbbox)[mask_inbbox]
            step_id = torch.arange(mask_inbbox.shape[1]).view(1,-1).expand_as(mask_inbbox)[mask_inbbox]
        return ray_pts, ray_id, step_id, N_samples
    
    # new add our conditional flow structure
    def flow(self, h_feature, flag):

        BN = h_feature.shape[0]
        
        if flag == 'density':

            params = h_feature

            den_glob, den_glob_std = self.global_flow_den[:1], self.global_flow_den[1:]
            den_glob_std = F.softplus(den_glob_std) + 1e-3
            
            den_glob = den_glob[None,None].expand([BN,self.K_samples,1])
            den_glob_std = den_glob_std[None,None].expand([BN,self.K_samples,1])

            if not self.is_test:
                eps_den = torch.empty([self.K_samples,1]).normal_()
                eps_den = eps_den[None]
            else:
                eps_den = self.sample_den[None]
            den0 = eps_den.mul(den_glob_std).add_(den_glob).reshape([-1,1]) # (BNK,1)

            ## pass through flows
            # density
            params = params[:,None,:].expand([BN, self.K_samples, params.shape[-1]])
            params = params.reshape([-1, params.shape[-1]])
            z_den, sum_log_det_j_den = self.flows_den(den0, params) # (BxNxK, 1),  (BxNxK,)
            z_k_den = z_den.reshape([BN, self.K_samples, 1])
            sum_log_det_j_den = sum_log_det_j_den.reshape([BN, self.K_samples])

            ## add log_det_jacobian for the last activation functions
            # density softplus
            # assert sum_log_det_j_den.isnan().any() == False
            # sum_log_det_j_den += (z_k_den + self.act_shift).sum(-1) - F.softplus(z_k_den + self.act_shift).sum(-1)
            # assert sum_log_det_j_den.isnan().any() == False

            ## compute loss_entropy
            # density
            den0 = den0.reshape([BN, self.K_samples, 1])
            base_log_norm_den = -0.5 * den_glob_std.log()*2
            # assert base_log_norm_den.isnan().any() == False

            ## sum up loss_entropy
            loss_entropy = base_log_norm_den.mean(-1) - sum_log_det_j_den

            return z_k_den[...,0], loss_entropy

        elif flag == 'rgb':

            rgb_glob, rgb_glob_std = self.global_flow_rgb[:3], self.global_flow_rgb[3:]
            rgb_glob_std = F.softplus(rgb_glob_std) + 1e-3

            rgb_glob = rgb_glob[None,None].expand([BN,self.K_samples,3])
            rgb_glob_std = rgb_glob_std[None,None].expand([BN,self.K_samples,3])

            params = h_feature

            if not self.is_test:
                eps_rgb = torch.empty([self.K_samples,3]).normal_()
                eps_rgb = eps_rgb[None]
            else:
                eps_rgb = self.sample_rgb[None]
            rgb0 = eps_rgb.mul(rgb_glob_std).add_(rgb_glob).reshape([-1,3]) # (BxNxK,3)

            # rgb 
            params = params[:,None,:].expand([BN, self.K_samples, params.shape[-1]])
            params = params.reshape([-1, params.shape[-1]])
            z_rgb, sum_log_det_j_rgb = self.flows_rgb(rgb0, params) # (BxNxK, 3),  (BxNxK,)
            z_k_rgb = z_rgb.reshape([BN, self.K_samples, 3])
            sum_log_det_j_rgb = sum_log_det_j_rgb.reshape([BN, self.K_samples])
            
            # rgb
            # assert sum_log_det_j_rgb.isnan().any() == False
            # sum_log_det_j_rgb += z_k_rgb.sum(-1) - 2 * F.softplus(z_k_rgb).sum(-1)
            # assert sum_log_det_j_rgb.isnan().any() == False

            ## compute loss_entropy
            # rgb
            rgb0 = rgb0.reshape([BN, self.K_samples, 3]) 
            base_log_norm_rgb = -0.5 * rgb_glob_std.log()*2
            # assert base_log_norm_rgb.isnan().any() == False

            ## sum up loss_entropy
            loss_entropy = base_log_norm_rgb.mean(-1) - sum_log_det_j_rgb

            return z_k_rgb, loss_entropy

    def forward(self, rays_o, rays_d, viewdirs, global_step=1000, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        self.is_test = render_kwargs['is_test']

        # sample points on rays
        ray_pts, ray_id, step_id, N_samples = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for alpha w/ post-activation
        kd = self.density_params(ray_pts)
        raw_density_k, loss_entropy = self.flow(kd,'density') # (N,K)
        density_k = F.softplus(raw_density_k + self.act_shift(ray_pts)[:,None])
        alpha_k = self.activate_density(raw_density_k + self.act_shift(ray_pts)[:,None], interval)
        
        if self.fast_color_thres > 0:
            alpha = alpha_k.mean(-1)
            mask = (alpha > self.fast_color_thres)
            alpha_std = alpha_k.std(-1)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            alpha_k = alpha_k[mask]
            alpha_std = alpha_std[mask]
            density_k = density_k[mask]
            raw_density_k = raw_density_k[mask]
            loss_entropy = loss_entropy[mask]

        # compute accumulated transmittance
        shape = alpha_k.shape
        # alpha_std_sum = segment_coo(src=(alpha_std),index=ray_id,out=torch.zeros([N]),reduce='sum')
        # weights, alphainv_last = Alphas2Weights.apply(alpha_k.mean(-1), ray_id, N)
        # weights_k = weights[:,None].expand([weights.shape[0],self.K_samples])
        # alphainv_last_k = alphainv_last[:,None].expand([N,self.K_samples])
        weights_k, alphainv_last_k = Alphas2Weights_k.apply(alpha_k, ray_id, N)
        # weights_k, alphainv_last_k = Alphas2Weights_k.apply(alpha_k, ray_id, N, alpha_std.detach(), alpha_std_sum.detach())
        weights_k = weights_k.reshape(shape)
        alphainv_last_k = alphainv_last_k.reshape([N,self.K_samples])

        if self.fast_color_thres > 0:
            weights = weights_k.mean(-1)
            mask = (weights > self.fast_color_thres)
            weights_std = weights_k.std(-1)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            weights = weights[mask]
            weights_k = weights_k[mask]
            density_k = density_k[mask]
            raw_density_k = raw_density_k[mask]
            loss_entropy = loss_entropy[mask]

        # query for color
        vox_emb = self.k0(ray_pts)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(vox_emb)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb[ray_id]
            rgb_feat = torch.cat([vox_emb, viewdirs_emb], -1)
            raw_rgb = self.rgbnet(rgb_feat)
            
            # new pass through our flow 
            raw_rgb_k, loss_entropy_rgb = self.flow(raw_rgb,'rgb')
            loss_entropy += loss_entropy_rgb
            rgb_k = torch.sigmoid(raw_rgb_k) # (N,K,3)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights_k[...,None] * rgb_k),
                index=ray_id,
                out=torch.zeros([N, self.K_samples, 3]),
                reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and global_step is not None:
            rgb_marched += (alphainv_last_k.unsqueeze(-1) * torch.rand_like(rgb_marched))
        else:
            rgb_marched += (alphainv_last_k.unsqueeze(-1) * render_kwargs['bg'])
        s = (step_id+0.5) / N_samples
        ret_dict.update({
            'alphainv_last': alphainv_last_k.mean(-1),
            'alphainv_last_k': alphainv_last_k,
            'weights': weights_k.mean(-1),
            'weights_k': weights_k,
            'rgb_marched': rgb_marched,
            'density_k': density_k,
            'raw_rgb_k': rgb_k,
            'ray_id': ray_id,
            'n_max': N_samples,
            's': s,
            'loss_entropy': loss_entropy,
            'ray_pts': ray_pts,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth_k = segment_coo(
                        src=(weights_k * s.unsqueeze(-1)),
                        index=ray_id,
                        out=torch.zeros([N, self.K_samples]),
                        reduce='sum')
            ret_dict.update({'render_depth': depth_k})

        return ret_dict


class DirectMPIGO_flow(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, mpi_depth=0,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=0,
                 K_samples=12,
                 num_flows=[4,4],
                 density_std_init=1,
                 glob_lambda_den=0.01,
                 **kwargs):
        super(DirectMPIGO_flow, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        denfea_dim = num_flows[0] * 4
        rgbfea_dim = num_flows[1] * (3*3+3+3+3) + 6
        self.num_flows = num_flows
        self.density_std_init = density_std_init
        self.glob_lambda_den = glob_lambda_den
        self.glob_lambda_rgb = glob_lambda_den

        print('K_samples',K_samples)
        print('density_std_init',density_std_init)
        print('num_flows',num_flows)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels, mpi_depth)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
            density_type, channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            config=self.density_config)
        
        # init density std voxel grid
        self.density_std = grid.create_grid(
            density_type, channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            config=self.density_config)
        with torch.no_grad():
            for i in range(mpi_depth):
                self.density_std.grid[..., i].fill_(np.log(np.exp(density_std_init * i / mpi_depth+0.1) - 1))
        
        # init params voxel grid
        self.density_params = grid.create_grid(
            density_type, channels=denfea_dim, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max,
            config=self.density_config)
        with torch.no_grad():
            self.density_params.grid.data.normal_(0, 0.05)

        # init density bias so that the initial contribution (the alpha values)
        # of each query points on a ray is equal
        self.act_shift = grid.DenseGrid(
                channels=1, world_size=[1,1,mpi_depth],
                xyz_min=xyz_min, xyz_max=xyz_max)
        self.act_shift.grid.requires_grad = False
        with torch.no_grad():
            g = np.full([mpi_depth], 1./mpi_depth - 1e-6)
            p = [1-g[0]]
            for i in range(1, len(g)):
                p.append((1-g[:i+1].sum())/(1-g[:i].sum()))
            for i in range(len(p)):
                self.act_shift.grid[..., i].fill_(np.log(p[i] ** (-1/self.voxel_size_ratio) - 1) - 1)

        # init color representation
        # feature voxel grid + shallow MLP  (fine stage)
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2) + self.k0_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, rgbfea_dim),
            )
            # nn.init.constant_(self.rgbnet[-1].bias, 0)

        print('dmpigo: densitye grid', self.density)
        print('dmpigo: feature grid', self.k0)
        print('dmpigo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        
         # init for our add flow
        self.K_samples = K_samples

        self.global_flow_den = torch.nn.Parameter(torch.zeros(2))
        self.global_flow_rgb = torch.nn.Parameter(torch.zeros(6))

        self.sample_size = self.K_samples
        self.sample_den = torch.empty([self.sample_size,1]).normal_()
        self.sample_rgb = torch.empty([self.sample_size,3]).normal_()
    
        self.flows_den = TriangularSylvesterNeRF_Voxel(num_flows[0],'density')
        self.flows_rgb = TriangularSylvesterNeRF_Voxel(num_flows[1],'rgb')

    def _set_grid_resolution(self, num_voxels, mpi_depth):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256. / mpi_depth
        print('dmpigo: world_size      ', self.world_size)
        print('dmpigo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'mpi_depth': self.mpi_depth,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            'K_samples': self.K_samples,
            'num_flows': self.num_flows,
            'density_std_init': self.density_std_init,
            'glob_lambda_den': self.glob_lambda_den,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, mpi_depth):
        print('dmpigo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels, mpi_depth)
        print('dmpigo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)
        self.density_std.scale_volume_grid(self.world_size)
        self.density_params.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            
            cache_grid_density = self.density(self_grid_xyz) # (*world_size)
            cache_grid_density_std = self.density_std(self_grid_xyz) # (*world_size)
            cache_grid_density_params = self.density_params(self_grid_xyz) # (*world_size)
            print('cache_grid_density',cache_grid_density.max(),cache_grid_density.min())
            print('cache_grid_density_std',cache_grid_density_std.max(),cache_grid_density_std.min())
            print('cache_grid_density_params',cache_grid_density_params.max(),cache_grid_density_params.min())
            params = cache_grid_density_params.view([-1,cache_grid_density_params.shape[-1]])
            den_glob = self.global_flow_den[0:1][None,None,:].expand_as(cache_grid_density)
            den_joint = cache_grid_density + den_glob * self.glob_lambda_den

            # den_std = F.softplus(cache_grid_density_std) + 1e-03
            # den_glob_std = self.global_flow_den[1:2][None,None,:].expand_as(den_std)
            # den_glob_std = F.softplus(den_glob_std) + 1e-03
            # den_joint_std = (den_std**2 + (den_glob_std * self.glob_lambda_rgb)**2).sqrt()
            # eps_den = torch.Tensor([3.])
            # den_joint = eps_den.mul(den_joint_std).add_(den_joint)

            den_flow, _ = self.flows_den(den_joint.reshape([-1,1]), params)
            den_flow = den_flow.reshape(cache_grid_density.shape) + self.act_shift.grid
            cache_grid_alpha = self.activate_density(den_flow)
            self_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)
            ori_p = self.mask_cache.mask.float().mean().item()
            print(f'dcvgo: mask_cache after scale {ori_p:.4f}')

            if ori_p < 0.75 and self.global_flow_den.requires_grad:
                print('set grad of global_flow_den to be False')
                self.global_flow_den.requires_grad = False
                # self.global_flow_rgb.requires_grad = False

        print('dmpigo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)

        cache_grid_density = self.density(cache_grid_xyz) # (*world_size)
        cache_grid_density_std = self.density_std(cache_grid_xyz) # (*world_size)
        cache_grid_density_params = self.density_params(cache_grid_xyz) # (*world_size)
        print('cache_grid_density',cache_grid_density.max(),cache_grid_density.min())
        print('cache_grid_density_std',cache_grid_density_std.max(),cache_grid_density_std.min())
        print('cache_grid_density_params',cache_grid_density_params.max(),cache_grid_density_params.min())
        params = cache_grid_density_params.view([-1,cache_grid_density_params.shape[-1]])
        den_glob = self.global_flow_den[0:1][None,None,:].expand_as(cache_grid_density)
        den_joint = cache_grid_density + den_glob * self.glob_lambda_den

        # den_std = F.softplus(cache_grid_density_std) + 1e-03
        # den_glob_std = self.global_flow_den[1:2][None,None,:].expand_as(den_std)
        # den_glob_std = F.softplus(den_glob_std) + 1e-03
        # den_joint_std = (den_std**2 + (den_glob_std * self.glob_lambda_rgb)**2).sqrt()
        # eps_den = torch.Tensor([3.])
        # den_joint = eps_den.mul(den_joint_std).add_(den_joint)

        den_flow, _ = self.flows_den(den_joint.reshape([-1,1]), params)
        den_flow = den_flow.reshape(cache_grid_density.shape) + self.act_shift.grid
        cache_grid_alpha = self.activate_density(den_flow)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dcvgo: update mask_cache {ori_p:.4f} => {new_p:.4f}')

        if new_p < 0.75 and self.global_flow_den.requires_grad:
            print('set grad of global_flow_den to be False')
            self.global_flow_den.requires_grad = False
            # self.global_flow_rgb.requires_grad = False

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz, render_kwargs, maskout_lt_nviews):
        print('dmpigo: update mask_cache lt_nviews start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, ray_id, step_id, N_samples = self.sample_ray(
                        rays_o=rays_o.to(device), rays_d=rays_d.to(device), **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += (ones.grid.grad > 1)
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0,0]
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')
        torch.cuda.empty_cache()
        eps_time = time.time() - eps_time
        print(f'dmpigo: update mask_cache lt_nviews finish (eps time:', eps_time, 'sec)')

    def density_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.density.total_variation_add_grad(wxy, wxy, wz, dense_mode)
    
    def params_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.density_std.total_variation_add_grad(wxy, wxy, wz, dense_mode)
        self.density_params.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        wxy = weight * self.world_size[:2].max() / 128
        wz = weight * self.mpi_depth / 128
        self.k0.total_variation_add_grad(wxy, wxy, wz, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), 0, interval).reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        assert near==0 and far==1
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N_samples = int((self.mpi_depth-1)/stepsize) + 1
        ray_pts, mask_outbbox = render_utils_cuda.sample_ndc_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, N_samples)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        if mask_inbbox.all():
            ray_id, step_id = create_full_step_id(mask_inbbox.shape)
        else:
            ray_id = torch.arange(mask_inbbox.shape[0]).view(-1,1).expand_as(mask_inbbox)[mask_inbbox]
            step_id = torch.arange(mask_inbbox.shape[1]).view(1,-1).expand_as(mask_inbbox)[mask_inbbox]
        return ray_pts, ray_id, step_id, N_samples
    
    # new add our conditional flow structure
    def flow(self, h_feature, flag):

        BN = h_feature.shape[0]
        
        if flag == 'density':
            
            den_mean, den_std = h_feature[...,0:1], h_feature[...,1:2]
            den_std = F.softplus(den_std) + 1e-3

            params = h_feature[...,2:]

            den_glob, den_glob_std = self.global_flow_den[:1], self.global_flow_den[1:]
            den_glob_std = F.softplus(den_glob_std) + 1e-3
            # den_glob, den_glob_std = self.global_flow_den[:1], torch.Tensor([1])

            # sample density, sum of coordinate-based distribution and global distribution, both are gaussian
            den_mean_k = den_mean[:,None,:].expand([BN, self.K_samples, 1]) + den_glob[None,None,:].expand([BN, self.K_samples, 1]) * self.glob_lambda_den
            den_std_k = (den_std[:,None,:].expand([BN, self.K_samples, 1])**2 + (den_glob_std[None,None,:] * self.glob_lambda_den).expand([BN, self.K_samples, 1])**2).sqrt()
            if not self.is_test:
                eps_den = torch.empty([self.K_samples,1]).normal_()
                eps_den = eps_den[None,...].expand_as(den_mean_k)
            else:
                eps_den = self.sample_den[None,...].expand_as(den_mean_k)
                # eps_den = torch.cat([eps_den[:,:-1,:], torch.zeros([BN,1,1]).to(den_mean_k.device)],-2)
            den0 = eps_den.mul(den_std_k).add_(den_mean_k).reshape([-1,1]) # (BN, K,1)

            ## pass through flows
            # density
            params = params[:,None,:].expand([BN, self.K_samples, params.shape[-1]])
            params = params.reshape([-1, params.shape[-1]])
            z_den, sum_log_det_j_den = self.flows_den(den0, params) # (BxNxK, 1),  (BxNxK,)
            z_k_den = z_den.reshape([BN, self.K_samples, 1])
            sum_log_det_j_den = sum_log_det_j_den.reshape([BN, self.K_samples])

            ## add log_det_jacobian for the last activation functions
            # density softplus
            # assert sum_log_det_j_den.isnan().any() == False
            # sum_log_det_j_den += (z_k_den + self.act_shift).sum(-1) - F.softplus(z_k_den + self.act_shift).sum(-1)
            # assert sum_log_det_j_den.isnan().any() == False

            ## compute loss_entropy
            # density
            den0 = den0.reshape([BN, self.K_samples, 1])
            base_log_norm_den = -0.5 * (den_std_k.log()*2 + (den0 - den_mean_k) * (den0 - den_mean_k) * (den_std_k**2).reciprocal())
            # assert base_log_norm_den.isnan().any() == False

            ## sum up loss_entropy
            loss_entropy = base_log_norm_den.mean(-1) - sum_log_det_j_den

            return z_k_den[...,0], loss_entropy

        elif flag == 'rgb':

            rgb_glob, rgb_glob_std = self.global_flow_rgb[:3], self.global_flow_rgb[3:]
            rgb_glob_std = F.softplus(rgb_glob_std) + 1e-3

            rgb_mean, rgb_std = h_feature[...,:3], h_feature[...,3:6]
            rgb_std = F.softplus(rgb_std) + 1e-3
            params = h_feature[...,6:]

            # sample rgb, sum of coordinate-based distribution and global distribution, both are gaussian
            rgb_mean_k = rgb_mean[:,None,:].expand([BN, self.K_samples, 3]) + rgb_glob[None,None,:].expand([BN, self.K_samples, 3]) * self.glob_lambda_rgb
            rgb_std_k = (rgb_std[:,None,:]**2 + (rgb_glob_std[None,None,:]*self.glob_lambda_rgb)**2).sqrt().expand([BN, self.K_samples, 3])
            if not self.is_test:
                eps_rgb = torch.empty([self.K_samples,3]).normal_()
                eps_rgb = eps_rgb[None,...].expand_as(rgb_mean_k)
            else:
                eps_rgb = self.sample_rgb[None,...].expand_as(rgb_mean_k)
                # eps_rgb = torch.cat([eps_rgb[:,:-1,:], torch.zeros([BN,1,3]).to(rgb_mean_k.device)],-2)
            rgb0 = eps_rgb.mul(rgb_std_k).add_(rgb_mean_k).reshape([-1,3]) # (BxNxK,3)

            # rgb 
            params = params[:,None,:].expand([BN, self.K_samples, params.shape[-1]])
            params = params.reshape([-1, params.shape[-1]])
            z_rgb, sum_log_det_j_rgb = self.flows_rgb(rgb0, params) # (BxNxK, 3),  (BxNxK,)
            z_k_rgb = z_rgb.reshape([BN, self.K_samples, 3])
            sum_log_det_j_rgb = sum_log_det_j_rgb.reshape([BN, self.K_samples])
            
            # rgb
            # assert sum_log_det_j_rgb.isnan().any() == False
            # sum_log_det_j_rgb += z_k_rgb.sum(-1) - 2 * F.softplus(z_k_rgb).sum(-1)
            # assert sum_log_det_j_rgb.isnan().any() == False

            ## compute loss_entropy
            # rgb
            rgb0 = rgb0.reshape([BN, self.K_samples, 3]) 
            base_log_norm_rgb = -0.5 * (rgb_std_k.log()*2 + (rgb0 - rgb_mean_k) * (rgb0 - rgb_mean_k) * (rgb_std_k**2).reciprocal())
            # assert base_log_norm_rgb.isnan().any() == False

            ## sum up loss_entropy
            loss_entropy = base_log_norm_rgb.mean(-1) - sum_log_det_j_rgb

            return z_k_rgb, loss_entropy

    def forward(self, rays_o, rays_d, viewdirs, global_step=1000, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        self.is_test = render_kwargs['is_test']

        # sample points on rays
        ray_pts, ray_id, step_id, N_samples = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for alpha w/ post-activation
        density_mean = self.density(ray_pts)
        density_std = self.density_std(ray_pts)
        density_params = self.density_params(ray_pts)
        kd = torch.cat([density_mean[:,None], density_std[:,None], density_params],-1)

        if global_step > 900:
            raw_density_k, loss_entropy = self.flow(kd,'density') # (N,K)
        else:
            BN = density_mean.shape[0]
            raw_density_k, loss_entropy = density_mean[:,None].expand([BN, self.K_samples]), torch.zeros_like(density_mean)
        density_k = F.softplus(raw_density_k + self.act_shift(ray_pts)[:,None])
        alpha_k = self.activate_density(raw_density_k + self.act_shift(ray_pts)[:,None], interval)
        
        if self.fast_color_thres > 0:
            alpha = alpha_k.mean(-1)
            mask = (alpha > self.fast_color_thres)
            alpha_std = alpha_k.std(-1)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            alpha_k = alpha_k[mask]
            alpha_std = alpha_std[mask]
            density_k = density_k[mask]
            raw_density_k = raw_density_k[mask]
            loss_entropy = loss_entropy[mask]

        # compute accumulated transmittance
        shape = alpha_k.shape
        # alpha_std_sum = segment_coo(src=(alpha_std),index=ray_id,out=torch.zeros([N]),reduce='sum')
        # weights, alphainv_last = Alphas2Weights.apply(alpha_k.mean(-1), ray_id, N)
        # weights_k = weights[:,None].expand([weights.shape[0],self.K_samples])
        # alphainv_last_k = alphainv_last[:,None].expand([N,self.K_samples])
        weights_k, alphainv_last_k = Alphas2Weights_k.apply(alpha_k, ray_id, N)
        # weights_k, alphainv_last_k = Alphas2Weights_k.apply(alpha_k, ray_id, N, alpha_std.detach(), alpha_std_sum.detach())
        weights_k = weights_k.reshape(shape)
        alphainv_last_k = alphainv_last_k.reshape([N,self.K_samples])

        if self.fast_color_thres > 0:
            weights = weights_k.mean(-1)
            mask = (weights > self.fast_color_thres)
            weights_std = weights_k.std(-1)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            weights = weights[mask]
            weights_k = weights_k[mask]
            density_k = density_k[mask]
            raw_density_k = raw_density_k[mask]
            loss_entropy = loss_entropy[mask]

        # query for color
        vox_emb = self.k0(ray_pts)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(vox_emb)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb[ray_id]
            rgb_feat = torch.cat([vox_emb, viewdirs_emb], -1)
            raw_rgb = self.rgbnet(rgb_feat)
            
            # new pass through our flow 
            if global_step > 900:
                raw_rgb_k, loss_entropy_rgb = self.flow(raw_rgb,'rgb')
            else:
                BN = raw_rgb.shape[0]
                raw_rgb_k, loss_entropy_rgb = raw_rgb[:,None,:3].expand([BN, self.K_samples, 3]), torch.zeros_like(raw_rgb[:,0])

            loss_entropy += loss_entropy_rgb
            rgb_k = torch.sigmoid(raw_rgb_k) # (N,K,3)

            # rgb_mean, rgb_std = raw_rgb[:,:3], raw_rgb[:,3:6]
            # rgb_std = F.softplus(rgb_std) + 1e-3
            # # rgb_std = torch.zeros_like(rgb_std)
            # if not render_kwargs['is_test']:
            #     eps_rgb = torch.empty([1,self.K_samples, 3]).normal_()
            # else:
            #     eps_rgb = self.sample_rgb[None]
            # raw_rgb_k = eps_rgb.mul(rgb_std[:,None,:]).add_(rgb_mean[:,None,:]) # (BN,K,3)
            # rgb_k = torch.sigmoid(raw_rgb_k)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights_k[...,None] * rgb_k),
                index=ray_id,
                out=torch.zeros([N, self.K_samples, 3]),
                reduce='sum')
        if render_kwargs.get('rand_bkgd', False) and global_step is not None:
            rgb_marched += (alphainv_last_k.unsqueeze(-1) * torch.rand_like(rgb_marched))
        else:
            rgb_marched += (alphainv_last_k.unsqueeze(-1) * render_kwargs['bg'])
        s = (step_id+0.5) / N_samples
        ret_dict.update({
            'alphainv_last': alphainv_last_k.mean(-1),
            'alphainv_last_k': alphainv_last_k,
            'weights': weights_k.mean(-1),
            'weights_k': weights_k,
            'rgb_marched': rgb_marched,
            'density_k': density_k,
            'raw_rgb_k': rgb_k,
            'ray_id': ray_id,
            'n_max': N_samples,
            's': s,
            'loss_entropy': loss_entropy,
            'ray_pts': ray_pts,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth_k = segment_coo(
                        src=(weights_k * s.unsqueeze(-1)),
                        index=ray_id,
                        out=torch.zeros([N, self.K_samples]),
                        reduce='sum')
            ret_dict.update({'render_depth': depth_k})

        return ret_dict


@functools.lru_cache(maxsize=128)
def create_full_step_id(shape):
    ray_id = torch.arange(shape[0]).view(-1,1).expand(shape).flatten()
    step_id = torch.arange(shape[1]).view(1,-1).expand(shape).flatten()
    return ray_id, step_id

