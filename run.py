
import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
import cv2, math

import mmengine
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, dcvgo, dmpigo, load_llff, grid
from lib.load_data import load_data
from utils.io_utils import *

from torch_efficient_distloss import flatten_eff_distloss

from skimage.metrics import structural_similarity as SSIM


torch.autograd.set_detect_anomaly(True)

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", action='store_true',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", action='store_true')
    parser.add_argument("--export_fine_only", action='store_true')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_depth", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_ause", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # uncertainty options
    parser.add_argument("--update_uncertainty_coarse", action='store_true')
    parser.add_argument("--update_uncertainty_fine", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')

    return parser

@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False, datadir=None,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0, eval_depth=False,
                      eval_ssim=False, eval_nll=False, eval_ause=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    
    rgbs, disps, depths = [],[],[]
    rgbs_std = []
    disps_std = []
    depths = []
    bgmaps = []
    ause_rmse = []
    ausc_rmse = []
    ause_mae = []
    ausc_mae = []
    psnrs, ssims, lpips_alex, lpips_vgg = [],[],[],[]

    auses_C, auses_VH, auses_CH, auses_epi = [],[],[],[]
    auscs_C, auscs_VH, auscs_CH, auscs_epi = [],[],[],[]

    w_sums = []

    t0 = time.time()

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        i_test = render_kwargs['i_test']
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)

        keys = ['rgb_marched', 'U_VH', 'U_epi', 'w_sum', 'render_depth']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(2048, 0), rays_d.split(2048, 0), viewdirs.split(2048, 0))
        ]
        if render_result_chunks[0]['rgb_marched'].ndim == 2:
            render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks])#.reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
            }
            rgb = render_result['rgb_marched'].cpu().numpy().reshape(H,W,-1)
            depth = render_result['render_depth'].cpu().numpy().reshape(H,W,-1)

            w_sum = render_result['w_sum'].cpu().numpy().reshape(H,W,-1)
            w_sums.append(w_sum)

            print('forward time:',time.time()-t0)
    
            disp = 1 - depth / np.max(depth)
        
            rgbs.append(rgb)
            disps.append(disp)
            depths.append(depth)

            U_VH = render_result['U_VH'].cpu().numpy().reshape(H,W,-1)
            U_VH = (1-np.exp(-(U_VH)))**2
            print('U_VH',U_VH.max(),U_VH.min())

            U_epi = (1-w_sum)**2
            print('U_epi',U_epi.max(),U_epi.min())

            # # imwrite U_VH
            savedir_U_VH = os.path.join(savedir, 'U_VH')
            os.makedirs(savedir_U_VH, exist_ok=True)
            U_VH_norm = utils.NormalizeData(U_VH)
            U_VH_8 = utils.to8b(U_VH_norm)
            heatmap_U_VH = cv2.applyColorMap(U_VH_8, cv2.COLORMAP_MAGMA)
            filename = os.path.join(savedir_U_VH, '{:03d}_U_VH.png'.format(i_test[i]))
            cv2.imwrite(filename, heatmap_U_VH)

            # imwrite U_epi + rgb_k_std
            U_epi = U_epi
            savedir_U_epi = os.path.join(savedir, 'rgb', 'U_epi')
            os.makedirs(savedir_U_epi, exist_ok=True)
            U_epi_norm = utils.NormalizeData(U_epi)
            U_epi_8 = utils.to8b(U_epi_norm)
            heatmap_U_epi = cv2.applyColorMap(U_epi_8, cv2.COLORMAP_MAGMA)
            filename = os.path.join(savedir_U_epi, '{:03d}_U_epi.png'.format(i_test[i]))
            cv2.imwrite(filename, heatmap_U_epi)
            
            error = (rgb - gt_imgs[i])**2
            unc = U_VH
            ause, ausc = utils.compute_ause(i_test[i], error, unc, savedir, 'VH')
            auscs_VH.append(ausc)
            auses_VH.append(ause)

            unc = U_epi
            ause, ausc = utils.compute_ause(i_test[i], error, unc, savedir, 'epi')
            auscs_epi.append(ausc)
            auses_epi.append(ause)

            if i==0:
                print('Testing', rgb.shape)

            if gt_imgs is not None and render_factor==0:
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                print('psnr:',p)
                psnrs.append(p)
                if eval_ssim:
                    ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
                if eval_lpips_alex:
                    lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
                if eval_lpips_vgg:
                    lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))
        else:
            k_samples = render_result_chunks[0]['rgb_marched'].shape[1]
            render_result = {
                k: torch.cat([ret[k] for ret in render_result_chunks])
                for k in render_result_chunks[0].keys()
            }
            rgb_k = render_result['rgb_marched'].cpu().numpy().reshape(H,W,k_samples,-1)

            print('forward time:',time.time()-t0)

            rgb_k_mean = rgb_k.mean(-2)
            rgb_k_std = rgb_k.std(-2)

            rgbs.append(rgb_k_mean)
            rgbs_std.append(rgb_k_std)

            U_VH = render_result['U_VH'].cpu().numpy().reshape(H,W,-1)
            U_epi = render_result['U_epi'].cpu().numpy().reshape(H,W,-1)
            print('U_epi',U_epi.max(),U_epi.min())
    
            U_VH = (1-np.exp(-(U_VH)))**2
            print('U_VH',U_VH.max(),U_VH.min())

            # imwrite U_VH
            savedir_U_VH = os.path.join(savedir, 'U_VH')
            os.makedirs(savedir_U_VH, exist_ok=True)
            U_VH_norm = utils.NormalizeData(U_VH)
            U_VH_8 = utils.to8b(U_VH_norm)
            heatmap_U_VH = cv2.applyColorMap(U_VH_8, cv2.COLORMAP_MAGMA)
            filename = os.path.join(savedir_U_VH, '{:03d}_U_VH.png'.format(i_test[i]))
            cv2.imwrite(filename, heatmap_U_VH)

            # imwrite U_epi + rgb_k_std
            U_epi = U_epi + rgb_k_std
            savedir_U_epi = os.path.join(savedir, 'rgb', 'U_epi')
            os.makedirs(savedir_U_epi, exist_ok=True)
            U_epi_norm = utils.NormalizeData(U_epi)
            U_epi_8 = utils.to8b(U_epi_norm)
            heatmap_U_epi = cv2.applyColorMap(U_epi_8, cv2.COLORMAP_MAGMA)
            filename = os.path.join(savedir_U_epi, '{:03d}_U_epi.png'.format(i_test[i]))
            cv2.imwrite(filename, heatmap_U_epi)

            # imwrite U_VH + rgb_k_std
            Unc = U_VH + rgb_k_std
            savedir_U_VH = os.path.join(savedir, 'rgb','Uncertainty')
            os.makedirs(savedir_U_VH, exist_ok=True)
            Unc = utils.NormalizeData(Unc)
            U_VH_8 = utils.to8b(Unc)
            heatmap_U_VH = cv2.applyColorMap(U_VH_8, cv2.COLORMAP_MAGMA)
            filename = os.path.join(savedir_U_VH, '{:03d}_U.png'.format(i_test[i]))
            cv2.imwrite(filename, heatmap_U_VH)

            if gt_imgs is not None and render_factor==0:
                p = -10. * np.log10(np.mean(np.square(rgb_k_mean - gt_imgs[i])))
                print('psnr:',p)
                psnrs.append(p)
                if eval_ause:
                    ause, ausc = utils.rgb_ause(i, rgb_k, gt_imgs[i], savedir, type='rmse')
                    ause_rmse.append(ause)
                    ausc_rmse.append(ausc)
                    ause, ausc = utils.rgb_ause(i, rgb_k, gt_imgs[i], savedir, type='mae')
                    ause_mae.append(ause)
                    ausc_mae.append(ausc)
                if eval_ssim:
                    # ssims.append(utils.rgb_ssim(rgb_k_mean, gt_imgs[i], max_val=1))
                    ssim = SSIM(rgb_k_mean, gt_imgs[i], multichannel=True)
                    ssims.append(ssim)
                if eval_lpips_alex:
                    lpips_alex.append(utils.rgb_lpips(rgb_k_mean, gt_imgs[i], net_name='alex', device=c2w.device))
                if eval_lpips_vgg:
                    lpips_vgg.append(utils.rgb_lpips(rgb_k_mean, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
    if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
    if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
    if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')
    
    # if len(auses_VH):
    #     print('AUSC with rgb_k_std:',np.mean(auscs_C), '(avg)')
    #     print('AUSC with VH:',np.mean(auscs_VH), '(avg)')
    #     print('AUSC with rgb_k_std and VH:',np.mean(auscs_CH), '(avg)')
    #     print('AUSC with rgb_k_std and U_epi:',np.mean(auscs_epi), '(avg)')
    #     print('AUSE with rgb_k_std:',np.mean(auses_C), '(avg)')
    #     print('AUSE with VH:',np.mean(auses_VH), '(avg)')
    #     print('AUSE with rgb_k_std and VH:',np.mean(auses_CH), '(avg)')
    #     print('AUSE with rgb_k_std and U_epi:',np.mean(auses_epi), '(avg)')
    
    # if len(ause_rmse):
    #     print('Testing ause rmse', np.mean(ause_rmse), '(avg)')
    #     print('Testing ause mae', np.mean(ause_mae), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            savedir_rgb = os.path.join(savedir, 'rgb')
            os.makedirs(savedir_rgb, exist_ok=True)
            filename = os.path.join(savedir_rgb, '{:03d}.png'.format(i_test[i]))
            imageio.imwrite(filename, rgb8)

            rgb_mae = np.abs(rgbs[i]-gt_imgs[i])
            rgb_mae = utils.NormalizeData(rgb_mae)
            rgb8_mae = utils.to8b(rgb_mae)
            heatmap_mae = cv2.applyColorMap(rgb8_mae, cv2.COLORMAP_MAGMA)
            savedir_rgbmae = os.path.join(savedir, 'rgbmae')
            os.makedirs(savedir_rgbmae, exist_ok=True)
            filename = os.path.join(savedir_rgbmae, '{:03d}_rgbmae.png'.format(i_test[i]))
            cv2.imwrite(filename, heatmap_mae)

            rgb8_gt = utils.to8b(gt_imgs[i])
            savedir_gt = os.path.join(savedir, 'gt')
            os.makedirs(savedir_gt, exist_ok=True)
            filename = os.path.join(savedir_gt, '{:03d}_gt.png'.format(i_test[i]))
            imageio.imwrite(filename, rgb8_gt)
        
        imageio.mimwrite(os.path.join(savedir, 'video_rgb.mp4'), utils.to8b(rgbs), fps=2, quality=8)

        if len(rgbs_std):
            for i in trange(len(rgbs_std)):
                rgb_std = utils.NormalizeData(rgbs_std[i])
                rgb8_std = utils.to8b(rgb_std)
                heatmap_std = cv2.applyColorMap(rgb8_std, cv2.COLORMAP_MAGMA)
                savedir_rgbstd = os.path.join(savedir, 'rgbstd')
                os.makedirs(savedir_rgbstd, exist_ok=True)
                filename = os.path.join(savedir_rgbstd, '{:03d}_rgbstd.png'.format(i_test[i]))
                cv2.imwrite(filename, heatmap_std)
            imageio.mimwrite(os.path.join(savedir, 'video_rgbstd.mp4'), utils.to8b(rgbs_std), fps=2, quality=8)
        
    rgbs = np.array(rgbs)

    return rgbs

@torch.no_grad()
def update_uncertainty_grid(model, render_poses, HW, Ks, ndc, datatype, cfg, render_kwargs):
    
    path_VH_step1 = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname,'VH_step1.npz')
    if datatype == 'unbounded':
        if not os.path.exists(path_VH_step1):
            # iteratively go through all rays in training views
            # set unsampled voxels in the non-contracted space to be of zero uncertainty
            for i, c2w in enumerate(tqdm(render_poses)):

                H, W = HW[i]
                K = Ks[i]
                c2w = torch.Tensor(c2w)
                rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                        H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                        flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
                
                rays_o = rays_o.flatten(0,-2)
                rays_d = rays_d.flatten(0,-2)
                viewdirs = viewdirs.flatten(0,-2)
                chunk = 8192 * 20
                for ro, rd, vd in zip(rays_o.split(chunk, 0), rays_d.split(chunk, 0), viewdirs.split(chunk, 0)):
                    model.set_zero_unseen_foreground(ro, rd, vd, **render_kwargs)

            model.uncertainty_grid.update_indices_unseen_foreground(model.world_size)
            print('! Updated unaffected region uncertainty grid',(model.uncertainty_grid.grid.data==0).sum()/torch.prod(model.world_size))

            print('saving uncertainty_VH step1')
            uncertainty_VH = model.uncertainty_grid.get_dense_grid().cpu().numpy()
            np.savez_compressed(path_VH_step1, uncertainty_VH=uncertainty_VH)
        else:
            print('loading uncertainty_VH step1')
            uncertainty_VH = np.load(path_VH_step1)['uncertainty_VH']
            model.uncertainty_grid.grid = nn.Parameter(torch.tensor(uncertainty_VH))
    
    path_VH_step2 = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname,'VH_step2.npz')
    if not os.path.exists(path_VH_step2):
        # set occluded and outside voxels to be of high uncertainty
        for i, c2w in enumerate(tqdm(render_poses)):

            H, W = HW[i]
            K = Ks[i]
            c2w = torch.Tensor(c2w)
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
            
            rays_o = rays_o.flatten(0,-2)
            rays_d = rays_d.flatten(0,-2)
            viewdirs = viewdirs.flatten(0,-2)
            chunk = 8192
            for ro, rd, vd in zip(rays_o.split(chunk, 0), rays_d.split(chunk, 0), viewdirs.split(chunk, 0)):
                model.update_uncertainty(ro, rd, vd, **render_kwargs)
        
        model.uncertainty_grid.update_indices(model.world_size)
        print('! Finished updating uncertainty grid:',(model.uncertainty_grid.grid.data==0).sum()/torch.prod(model.world_size))

        print('saving uncertainty_VH step2')
        uncertainty_VH = model.uncertainty_grid.get_dense_grid().cpu().numpy()
        np.savez_compressed(path_VH_step2, uncertainty_VH=uncertainty_VH)
    else:
        print('loading uncertainty_VH step2')
        uncertainty_VH = np.load(path_VH_step2)['uncertainty_VH']
        model.uncertainty_grid.grid = nn.Parameter(torch.tensor(uncertainty_VH))

def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'i_train', 'i_all', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images', 'sc'}

    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)
    
    print('training views:',data_dict['i_train'])
    print('testing views:',data_dict['i_test'])

    return data_dict


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    # for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
    for (H, W), K, c2w in zip(HW, Ks, poses):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip 
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    print('center',center,radius)
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))

    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max

def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
        
    if cfg.data.unbounded_inward:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model_class = dcvgo.DirectContractedVoxGO_Sto
        model = model_class(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        if cfg_train.uncertainty:
            model_class = dvgo.DirectVoxGO_Sto
        else:
            model_class = dvgo.DirectVoxGO
        model = model_class(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)
    # model = nn.DataParallel(model).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer

def load_existed_model(args, cfg, cfg_train, reload_ckpt_path):
    if cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO_Sto
    else:
        if cfg_train.uncertainty:
            model_class = dvgo.DirectVoxGO_Sto
        else:
            model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start

def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_all, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_all', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model, optimizer = create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path)
        
    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'near_clip': data_dict['near_clip'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'is_train': True,
    }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, (dvgo.DirectVoxGO, dcvgo.DirectContractedVoxGO)):
                model.scale_volume_grid(cur_voxels)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.act_shift -= cfg_train.decay_after_scale
            torch.cuda.empty_cache()

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result = model(
            rays_o, rays_d, viewdirs,
            global_step=global_step, N_iters=cfg_train.N_iters, N_rand=cfg_train.N_rand,
            **render_kwargs)

        rgb = render_result['rgb_marched']
        raw_rgb = render_result['raw_rgb']
        alphainv_last = render_result['alphainv_last']

        weights = render_result['weights']
        ray_id = render_result['ray_id']

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(rgb, target)
        psnr = utils.mse2psnr(loss.detach())
        if cfg_train.weight_entropy_last > 0:
            pout = alphainv_last.clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_nearclip > 0:
            t = render_result['t']
            raw_density = render_result['raw_density']
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (t < near_thres)
            density = raw_density[near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            if len(weights):
                loss_distortion = flatten_eff_distloss(weights, s, 1/n_max, ray_id)
                loss += cfg_train.weight_distortion * loss_distortion
        if cfg_train.weight_rgbper > 0:
            rgbper = (raw_rgb - target[ray_id]).pow(2).sum(-1)
            rgbper_loss = (rgbper * weights.detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor
            
        if global_step%500==0:
            # print('fast_color_thres:',model.fast_color_thres)
            if 'ray_pts' in render_result:
                ray_pts = render_result['ray_pts']
                print('ray_pts:',ray_pts.shape)

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                    #    f'entropy_last_loss:{entropy_last_loss.item():.3f} / '
                    #    f'loss_distortion:{loss_distortion.item():.3f} / '
                    #    f'rgbper_loss:{rgbper_loss.item():.3f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)
    
    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)
    
    # construct an uncertain voxel grid with initialized high uncertainty
    if args.update_uncertainty_coarse:
        model_class = utils.select_model(cfg,cfg.coarse_train)
        model.uncertainty_grid.reset(model.world_size, cfg.data.datatype)
        update_uncertainty_grid(model, poses[i_train], HW[i_train], Ks[i_train], cfg.data.ndc, cfg.data.datatype, render_kwargs)
        torch.save({
            'global_step': max(global_step,start),
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)

        # filter out uncertainty grid outside bbox [xyz_fine_min, xyz_fine_max]
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, f'coarse_last.tar')
        model_class = utils.select_model(cfg,cfg.coarse_train)
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=model_class, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres)
        model.uncertainty_grid.maskout(xyz_min_fine, xyz_max_fine)
        print('mask out uncertainty grid outside bbox (xyz_fine_min, xyz_fine_max)',(self.grid.data==0).sum()/torch.prod(model.world_size))
        torch.save({
            'global_step': start,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)

def scene_rep_reconstruction_uncertainty(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):
    # init 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_all, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_all', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model, optimizer = create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'near_clip': data_dict['near_clip'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'is_test': False,
        'is_train': True,
    }
    
    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()
        
    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    print_time = []
    print_psnr = []
    time0 = time.time()
    global_step = -1
    adding_rays = False
    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, (dvgo.DirectVoxGO_Sto, dcvgo.DirectContractedVoxGO_Sto)):
                model.scale_volume_grid(cur_voxels)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.act_shift -= cfg_train.decay_after_scale
            torch.cuda.empty_cache()

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            # target_depth = depth_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]

        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]

        else:
            raise NotImplementedError
        
        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result = model(
            rays_o, rays_d, viewdirs,
            global_step=global_step, N_iters=cfg_train.N_iters, N_rand=cfg_train.N_rand,
            **render_kwargs)
        
        rgb_k = render_result['rgb_marched']
        ray_id = render_result['ray_id']

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)

        # new loss
        ############################# Multivariate kernel density estimation
        # compute mean and variance (N_rays, k, 3)
        if rgb_k.ndim == 3:
            rgb_mean = torch.mean(rgb_k,-2) # (N_rays, 3)
        else:
            rgb_mean = rgb_k
        mse_train = F.mse_loss(rgb_mean, target)
        psnr_train = utils.mse2psnr(mse_train.detach())
        
        if rgb_k.ndim == 3 and rgb_k.shape[1] != 1:
            eps = 1e-5
            k = rgb_k.shape[-2]
            rgb_std = torch.std(rgb_k, -2) * k / (k-1) # (N_rays, 3) 

            H_sqrt = rgb_std.detach() * torch.pow(0.8/k,torch.tensor(-1/7)) + eps # (N_rays, 3)
            H_sqrt = H_sqrt[:,None,:] # (N_rays, 1, 3)
            r_P_C_1 = torch.exp( -((rgb_k - target[:,None,:])**2) / (2*H_sqrt*H_sqrt)) # [N_rays, k, 3]
            r_P_C_2 = torch.pow(torch.tensor(2*math.pi),-1.5) / H_sqrt # [N_rays, 1, 3]
            r_P_C = r_P_C_1 * r_P_C_2 # [N_rays, k, 3]
            r_P_C_mean = r_P_C.mean(-2) + eps # [N_rays, 3]
            loss_nll = - torch.log(r_P_C_mean).mean() * cfg_train.weight_main

            loss = loss_nll

            loss_entropy_rgb = render_result['loss_entropy_rgb']
            loss_entropy_rgb = loss_entropy_rgb.mean() * cfg_train.weight_entropy_rgb
            loss += loss_entropy_rgb
            
            loss_entropy_den = render_result['loss_entropy_den']
            loss_entropy_den = loss_entropy_den.mean() * cfg_train.weight_entropy_den
            loss += loss_entropy_den
            
            assert loss_nll.isnan().any() == False
        else:
            loss = cfg_train.weight_main * mse_train

        if cfg_train.weight_entropy_last > 0:
            alphainv_last_k = render_result['alphainv_last_k']
            alphainv_last = render_result['alphainv_last']
            pout = alphainv_last.clamp(1e-6, 1-1e-6) 
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss 
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            weights_k = render_result['weights_k']
            weights = render_result['weights']
            if len(weights):
                loss_distortion = flatten_eff_distloss(weights, s, 1/n_max, ray_id)
                loss += cfg_train.weight_distortion * loss_distortion
        if cfg_train.weight_nearclip > 0:
            t = render_result['t']
            raw_density_k = render_result['raw_density_k']
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (t < near_thres)
            density_k = raw_density_k[near_mask]
            if len(density_k):
                nearclip_loss = (density_k - density_k.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        if cfg_train.weight_rgbper > 0:
            weights_k = render_result['weights_k']
            raw_rgb_k = render_result['raw_rgb_k']
            rgbper = (raw_rgb_k - target[ray_id][:,None]).pow(2).sum(-1)
            rgbper_loss = (rgbper * weights_k.detach()).sum() / len(rays_o) / raw_rgb_k.shape[1]
            loss += cfg_train.weight_rgbper * rgbper_loss
        if cfg_train.weight_density > 0:
            raw_density_k = render_result['raw_density_k']
            density_re_loss = (raw_density_k - raw_density_k.detach()).mean()
            loss += cfg_train.weight_density * density_re_loss
        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        psnr_lst.append(psnr_train.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # debug
        if global_step%500==0 or global_step==1:
            print('weight_entropy_den:',cfg_train.weight_entropy_den)
            print('weight_entropy_rgb:',cfg_train.weight_entropy_rgb)
        
        if global_step%500==0:
            ray_pts = render_result['ray_pts']
            print('ray_pts:',ray_pts.shape)
        
        if global_step%10==0:
            print_time.append(time.time() - time0)
            print_psnr.append(psnr_train.item())
                
        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.3f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                    #    f'entropy_last_loss:{entropy_last_loss.item():.3f} / '
                    #    f'loss_distortion:{loss_distortion.item():.3f} / '
                    #    f'rgbper_loss:{rgbper_loss.item():.3f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)
    
    # update the 3D uncertain voxel grid after training
    if args.update_uncertainty_fine:
        model.uncertainty_grid.reset(model.world_size, cfg.data.datatype)
        update_uncertainty_grid(model, poses[i_train], HW[i_train], Ks[i_train], cfg.data.ndc, cfg.data.datatype, cfg, render_kwargs)
        torch.save({
            'global_step': max(global_step,start),
            # 'i_train':i_train,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)

def train(args, cfg, data_dict):

    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, 'config.py'))

    # coarse geometry searching (only works for inward bounded scenes)
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    if cfg.coarse_train.N_iters > 0:
        if cfg.coarse_train.uncertainty:
            scene_rep_reconstruction_uncertainty(
                    args=args, cfg=cfg,
                    cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                    xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                    data_dict=data_dict, stage='coarse')
        else:
            scene_rep_reconstruction(
                    args=args, cfg=cfg,
                    cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                    xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                    data_dict=data_dict, stage='coarse')
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        if cfg.coarse_train.uncertainty:
            model_class=dvgo.DirectVoxGO_Sto
        else:
            model_class=dvgo.DirectVoxGO
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=model_class, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres)

    if cfg.fine_train.N_iters > 0:
        if cfg.fine_train.uncertainty:
            scene_rep_reconstruction_uncertainty(
                    args=args, cfg=cfg,
                    cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
                    xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
                    data_dict=data_dict, stage='fine',
                    coarse_ckpt_path=coarse_ckpt_path) # TODO: not suitable for our unknown space assigned with high uncertainty
        else:
            scene_rep_reconstruction(
                    args=args, cfg=cfg,
                    cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
                    xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
                    data_dict=data_dict, stage='fine',
                    coarse_ckpt_path=coarse_ckpt_path)
        eps_fine = time.time() - eps_fine
        eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
        print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmengine.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # our sampling process cannot be deterministic !!!
    # seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # train nerf
    if not args.render_only:
        train(args, cfg, data_dict)

    # export scene bbox and camera poses in 3d for debugging and visualization
    # dataname = 'africa'
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        testsavedir = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname)
        os.makedirs(testsavedir, exist_ok=True)
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train, i_test, i_all = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train'], data_dict['i_test'], data_dict['i_all']
        near, far = data_dict['near'], data_dict['far']
        if data_dict['near_clip'] is not None:
            near = data_dict['near_clip']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(testsavedir+'/cam_train.npz',
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        # save test cams
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_test], HW[i_test], Ks[i_test]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(testsavedir+'/cam_test.npz',
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        # save all cams
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_all], HW[i_all], Ks[i_all]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(testsavedir+'/cam_all.npz',
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')

    if args.export_coarse_only:
        print('Export coarse visualization...')
        ckpt_path = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, 'coarse_last.tar')
        testsavedir = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname)
        if cfg.coarse_train.uncertainty:
            with torch.no_grad():
                model_class = utils.select_model(cfg,cfg.coarse_train)
                model = utils.load_model(model_class, ckpt_path).to(device)
                density = model.density.get_dense_grid().squeeze().cpu().numpy()
                density_std = model.density_std.get_dense_grid().squeeze().cpu().numpy()
                uncertainty_VH = model.uncertainty_grid.get_dense_grid().squeeze().cpu().numpy()
                alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
                rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
                xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                                                model_class=dvgo.DirectVoxGO_Sto, model_path=ckpt_path,
                                                thres=cfg.fine_model_and_render.bbox_thres)
            np.savez_compressed(testsavedir+'/coarse.npz', density=density, density_std=density_std, uncertainty_VH=uncertainty_VH, alpha=alpha, rgb=rgb, xyz_min_fine=xyz_min_fine.cpu().numpy(), xyz_max_fine=xyz_max_fine.cpu().numpy())
        else:
            with torch.no_grad():
                model_class = utils.select_model(cfg,cfg.coarse_train)
                model = utils.load_model(model_class, ckpt_path).to(device)
                density = model.density.get_dense_grid().squeeze().cpu().numpy()
                uncertainty_VH = model.uncertainty_grid.get_dense_grid().squeeze().cpu().numpy()
                alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
                rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
                xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                                                model_class=dvgo.DirectVoxGO, model_path=ckpt_path,
                                                thres=cfg.fine_model_and_render.bbox_thres)
            np.savez_compressed(testsavedir+'/coarse.npz', density=density, alpha=alpha, uncertainty_VH=uncertainty_VH, rgb=rgb,xyz_min_fine=xyz_min_fine.cpu().numpy(), xyz_max_fine=xyz_max_fine.cpu().numpy())
        print('done')
        # sys.exit()
    
    if args.export_fine_only:
        print('Export fine visualization...')
        ckpt_path = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, 'fine_last.tar')
        testsavedir = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname)
        if cfg.fine_train.uncertainty:
            with torch.no_grad():
                model_class = utils.select_model(cfg,cfg.fine_train)
                model = utils.load_model(model_class, ckpt_path).to(device)
                xyz_min_fine = model.xyz_min.cpu().numpy()
                xyz_max_fine = model.xyz_max.cpu().numpy()
                density = model.density.get_dense_grid().squeeze().cpu().numpy()
                uncertainty_VH = model.uncertainty_grid.get_dense_grid().squeeze().cpu().numpy()
                density_std = model.density_std.get_dense_grid().squeeze().cpu().numpy()
                alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
                rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
            np.savez_compressed(testsavedir+'/fine.npz', density=density, density_std=density_std, uncertainty_VH=uncertainty_VH, alpha=alpha, rgb=rgb, xyz_min_fine=xyz_min_fine, xyz_max_fine=xyz_max_fine)
        else:
            with torch.no_grad():
                model_class = utils.select_model(cfg,cfg.fine_train)
                model = utils.load_model(model_class, ckpt_path).to(device)
                xyz_min_fine = model.xyz_min.cpu().numpy()
                xyz_max_fine = model.xyz_max.cpu().numpy()
                density = model.density.get_dense_grid().squeeze().cpu().numpy()
                uncertainty_VH = model.uncertainty_grid.get_dense_grid().squeeze().cpu().numpy()
                alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
                rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
            np.savez_compressed(testsavedir+'/fine.npz', alpha=alpha, rgb=rgb, density=density, uncertainty_VH=uncertainty_VH, xyz_min_fine=xyz_min_fine, xyz_max_fine=xyz_max_fine)
        print('done')
        # sys.exit()
    
    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model_class = utils.select_model(cfg,cfg.fine_train)
        print(model_class,ckpt_path)
        t0 = time.time()
        model = utils.load_model(model_class, ckpt_path).to(device)
        print('loading model time:', time.time()-t0)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'near_clip': data_dict['near_clip'],
                'far': data_dict['far'],
                'i_test': data_dict['i_test'],
                'i_train': data_dict['i_train'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
                'is_test': True,
                'compute_VH': True
            },
        }

    # render testset and eval
    if args.render_test:
        t0 = time.time()
        testsavedir = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir, dump_images=args.dump_images, datadir=cfg.data.datadir,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg, eval_ause=args.eval_ause,eval_depth=args.eval_depth,
                **render_viewpoints_kwargs)
        print('render_test time:',time.time()-t0)

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg, eval_ause=args.eval_ause,
                **render_viewpoints_kwargs)
        
    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.data.dataname, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)

    print('Done')