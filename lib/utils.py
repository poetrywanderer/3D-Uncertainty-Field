import os, math, cv2
import numpy as np
import scipy.signal
from typing import List, Optional
import matplotlib.pyplot as plt

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_adam import MaskedAdam

from . import dvgo, dcvgo, dmpigo


''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)

def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]
            
        if isinstance(model, nn.DataParallel):
            if not hasattr(model.module, k):
                continue
            param = getattr(model.module, k)
        else:
            if not hasattr(model, k):
                continue
            param = getattr(model, k)

        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return MaskedAdam(param_group)


''' Checkpoint utils
'''
def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer):
    ckpt = torch.load(ckpt_path)
    start = ckpt['global_step']
    model.load_state_dict(ckpt['model_state_dict'])
    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, optimizer, start


def load_model(model_class, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'])
    # for k,v in ckpt['model_state_dict'].items():
    #     print(k)
    return model 

def select_model(cfg, cfg_train):
    if cfg.data.unbounded_inward:
        if cfg_train.uncertainty:
            model_class = dcvgo.DirectContractedVoxGO_Sto
        else:
            model_class = dcvgo.DirectContractedVoxGO
    else:
        if cfg_train.uncertainty:
            model_class = dvgo.DirectVoxGO_Sto
        else:
            model_class = dvgo.DirectVoxGO
    
    return model_class


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()

def rgb_nll(rgb_k, rgb_std, gt_img):
    '''
    rgb_k: (H,W,K,3)
    '''
    eps = 1e-05
    n = rgb_k.shape[-2]
    # rgb_std = rgb_std * n / (n-1) # (H,W,3)

    # maskout rays intersecting with nothing
    # mask = (((1-rgb_k.mean(-2))+(1-gt_img)+rgb_std).mean(-1)==0)
    # rgb_std = rgb_std[~mask]
    # gt_img = gt_img[~mask]
    # rgb_k = rgb_k[~mask]

    # print('rgb_std,gt_img,rgb_k',rgb_std.shape,gt_img.shape,rgb_k.shape)
    # print('mask',mask.mean())

    # H_sqrt = rgb_std * np.power(0.8/n,-1/7) + eps # (H,W,3)
    # H_sqrt = H_sqrt[...,None,:] # (H,W,1,3)
    # r_P_C_1 = np.exp( -((rgb_k - gt_img[...,None,:])**2) / (2*H_sqrt*H_sqrt)) # (H,W,K,3)
    # r_P_C_2 = np.power(2*math.pi,-1.5) / H_sqrt # (H,W,1,3)
    # r_P_C = r_P_C_1 * r_P_C_2 # (H,W,K,3)
    # r_P_C_mean = r_P_C.mean(-2) + eps # (H,W,3)
    # nll = - np.log(r_P_C_mean)

    # mask = (rgb_std.mean(-1)==1)
    # print('mask',mask.shape)
    # rgb_std = rgb_std[mask]
    # gt_img = gt_img[mask]
    # rgb_k = rgb_k[mask]
    # print('rgb_std,gt_img,rgb_k',rgb_std.shape,gt_img.shape,rgb_k.shape)

    nll = 0.5 * np.log(rgb_std + eps) + 0.5 * (rgb_k.mean(-2) - gt_img)**2 / (rgb_std + eps)

    # Create a histogram
    # import matplotlib.pyplot as plt
    # plt.hist(nll.reshape(-1), bins=20, edgecolor='black')  # You can adjust the number of bins as needed
    # plt.xlabel('Negative Log-Likelihood')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Negative Log-Likelihood')
    # plt.grid(True)
    # plt.savefig('histogram_{:s}.png'.format(utype), dpi=300)  # You can adjust the filename and dpi as needed
    # plt.clf()
    # exit()

    # print('nll max min:',(-np.log(r_P_C_mean)).max(),(-np.log(r_P_C_mean)).min())

    return nll

def disp_nll(pred_k, pred_std, gt, utype='U_C'):
    '''
    pred_k: (H,W,K,1)
    pred_std, gt: (H,W,1)
    '''
    # eps = 1e-05
    # pred_std = pred_std[...,None] + eps
    # r_P_C_1 = np.exp( -((pred_k - gt[...,None])**2) / (2*pred_std*pred_std)) # [N_rays, 3, k]
    # r_P_C_2 = np.power(2*math.pi,-0.5) / pred_std # [N_rays, 3, 1]
    # r_P_C = r_P_C_1 * r_P_C_2 # [N_rays, 3, k]
    # r_P_C_mean = r_P_C.mean(-1) + eps
    # nll = - np.log(r_P_C_mean).mean()

    eps = 1e-05
    nll = 0.5 * np.log(pred_std + eps) + 0.5 * (pred_k.mean(-2) - gt)**2 / (pred_std + eps)
    nll = nll.mean()

    return nll

'''
copy from https://github.com/abdo-eldesokey/pncnn/blob/c6122e9c442eabeb0145b241121aeba0039eb5e7/utils/sparsification_plot.py#L10
'''
def sparsification_plot(var_vec, err_vec, uncert_type='c', err_type='rmse'):
    
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    # Sort the error
    #print('Sorting Error ...')
    err_vec_sorted, idxs = torch.sort(err_vec)
    # print('idxs',idxs)
    # exit()

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ause_err = []
    for r in ratio_removed:
        err_slice = err_vec_sorted[0:int((1-r)*n_valid_pixels)]
        if err_type == 'rmse':
            ause_err.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae':
            ause_err.append(err_slice.mean().cpu().numpy())

    # Normalize RMSE
    # print(ause_err[0])
    print('ause_err max:',ause_err[0])
    ause_err = ause_err / ause_err[0]
    ause_err = np.array(ause_err)

    ###########################################

    # Sort by variance
    #print('Sorting Variance ...')
    if uncert_type == 'c':
        var_vec = torch.sqrt(var_vec)
        _, var_vec_sorted_idxs = torch.sort(var_vec, descending=True)
    elif uncert_type == 'v':
        #var_vec = torch.exp(var_vec)
        var_vec = torch.sqrt(var_vec)
        _, var_vec_sorted_idxs = torch.sort(var_vec, descending=False)

    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]

    ause_err_by_var = []
    for r in ratio_removed:
        mse_err_slice = err_vec_sorted_by_var[0:int((1 - r) * n_valid_pixels)]
        if err_type == 'rmse':
            ause_err_by_var.append(torch.sqrt(mse_err_slice.mean()).cpu().numpy())
        elif err_type == 'mae':
            ause_err_by_var.append(mse_err_slice.mean().cpu().numpy())

    # Normalize RMSE
    print('ause_err_by_var max:',ause_err_by_var[0])
    ause_err_by_var = ause_err_by_var / ause_err_by_var[0]
    ause_err_by_var = np.array(ause_err_by_var)

    return ause_err, ause_err_by_var

def compute_ause(img_i, error, unc, savedir, unc_type='C'):
    '''
    rgb_k: (H,W,K,3)
    '''
    unc_r = unc.mean(-1).reshape(-1)
    error_r = error.mean(-1).reshape(-1)

    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    err_type = 'rmse'
    mse_by_err, mse_by_std = sparsification_plot(torch.tensor(unc_r), torch.tensor(error_r), uncert_type='v', err_type=err_type)
    savedir_ause = os.path.join(savedir, 'ause')
    os.makedirs(savedir_ause, exist_ok=True)
    ause = np.trapz(mse_by_std - mse_by_err, ratio_removed)
    ausc = np.trapz(mse_by_std, ratio_removed)
    print('- AUSC metric std - {:s} for {:s} is {:.5f}'.format(err_type, unc_type, ausc))
    print('- AUSE metric std - {:s} for {:s} is {:.5f}'.format(err_type, unc_type, ause))
    plt.clf()
    plt.plot(ratio_removed, mse_by_err, '--')
    plt.plot(ratio_removed, mse_by_std, '-r')
    plt.grid()
    plt.savefig(savedir_ause+'/{:s}_ause_{:s}_{:02d}.png'.format(err_type, unc_type, img_i))

    return ause, ausc

def rgb_ause(img_i, rgb_k, gt_img, savedir, type='rmse', rgb_or_depth='rgb'):
    '''
    rgb_k: (H,W,K,3)
    '''
    rgbs_mean = np.mean(rgb_k,-2) # (H,W,3)
    if type=='rmse':
        rgb_mse = (rgbs_mean-gt_img)**2
    elif type=='mae':
        rgb_mse = abs(rgbs_mean-gt_img)
    mse_r = np.mean(rgb_mse,-1).reshape(-1) # (N,)
    n = rgb_k.shape[-2]
    rgb_std = np.std(rgb_k, -2) * n / (n-1) # (N_rays, 3)
    std_r = np.mean(rgb_std,-1).reshape(-1) # (N,)
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    mse_by_err, mse_by_std = sparsification_plot(torch.tensor(std_r), torch.tensor(mse_r), uncert_type='v', err_type=type)
    if rgb_or_depth == 'rgb':
        savedir_ause = os.path.join(savedir, 'ause')
    else:
        savedir_ause = os.path.join(savedir, 'disp_ause')
    os.makedirs(savedir_ause, exist_ok=True)
    with open(savedir_ause + '/{:s}_by_error_{:02d}.npy'.format(type,img_i),'wb') as f:
        np.save(f,mse_by_err)
    with open(savedir_ause + '/{:s}_by_std_{:02d}.npy'.format(type,img_i),'wb') as f:
        np.save(f,mse_by_std)
    ause = np.trapz(mse_by_std - mse_by_err, ratio_removed)
    ausc = np.trapz(mse_by_std, ratio_removed)
    print('- AUSC metric std - {:s} is {:.5f}.'.format(type, ausc))
    print('- AUSE metric std - {:s} is {:.5f}.'.format(type, ause))
    plt.clf()
    plt.plot(ratio_removed, mse_by_err, '--')
    plt.plot(ratio_removed, mse_by_std, '-r')
    plt.grid()
    plt.savefig(savedir_ause+'/{:s}_ause_{:02d}.png'.format(type,img_i))

    return ause, ausc

def ray_parameterization(point3D, contracted_norm = 'inf'):
    '''
    ray_parameterization from mip-nerf
    '''
    bg_len = 1.0
    if contracted_norm == 'inf':
        norm = np.amax(abs(point3D), axis=-1, keepdims=True)
    elif contracted_norm == 'l2':
        norm = np.linalg.norm(point3D, axis=-1, keepdims=True)
    inner_mask = (norm<=1)
    point3D = np.where(
        inner_mask,
        point3D,
        point3D / norm * ((1+bg_len) - bg_len/norm)
    )

    return point3D


#######
# eval on depth
#######

def depth_eval(disps_k, disps_gt, disp_unc):
    '''
    disps_k: (H,W,K,1)
    disps_gt: (H,W,1)
    disp_unc: (H,W,1)
    '''
    disps_k = disps_k.squeeze()
    disps_gt = disps_gt.squeeze()
    disp_unc = disp_unc.squeeze()

    #### error evaluation ####
    disps_mean = np.mean(disps_k,-1) # (H,W)
    disp_rmse = np.sqrt(np.mean((disps_mean-disps_gt)**2))
    disp_mae = np.abs(disps_mean-disps_gt).mean()
    # delta 
    delta_thr1 = np.power(1.25,1)
    delta_thr2 = np.power(1.25,2)
    delta_thr3 = np.power(1.25,3)
    a = np.maximum(disps_mean.reshape(-1) / disps_gt.reshape(-1), disps_gt.reshape(-1) / disps_mean.reshape(-1))
    b1 = len(a[a < delta_thr1]) / len(a)
    b2 = len(a[a < delta_thr2]) / len(a)
    b3 = len(a[a < delta_thr3]) / len(a)

    #### uncertainty evaluation ####
    eps = 1e-05
    disp_unc = disp_unc[...,None] + eps
    r_P_C_1 = np.exp( -((disps_k - disps_gt[...,None])**2) / (2*disp_unc*disp_unc)) # [N_rays, 3, k]
    r_P_C_2 = np.power(2*math.pi,-0.5) / disp_unc # [N_rays, 3, 1]
    r_P_C = r_P_C_1 * r_P_C_2 # [N_rays, 3, k]
    r_P_C_mean = r_P_C.mean(-1) + eps
    nll = - np.log(r_P_C_mean).mean()

    # original depth nll
    # eps = 1e-05
    # n = disps_k.shape[-1]
    # disps_std = np.std(disps_k, -1) * n / (n-1) # (N_rays, 3)
    # H_sqrt = disps_std * np.power(0.8/n,-1/7) + eps # (N_rays, 3)
    # H_sqrt = H_sqrt[...,None] # (N_rays, 3, 1)
    # r_P_C_1 = np.exp( -((disps_k - disps_gt[...,None])**2) / (2*H_sqrt*H_sqrt)) # [N_rays, 3, k]
    # r_P_C_2 = np.power(2*math.pi,-1.5) / H_sqrt # [N_rays, 3, 1]
    # r_P_C = r_P_C_1 * r_P_C_2 # [N_rays, 3, k]
    # r_P_C_mean = r_P_C.mean(-1) + eps
    # nll = - np.log(r_P_C_mean).mean()

    return disp_rmse, disp_mae, b3, nll


def depth_ause(img_i, disp_k, disp_gt, disp_unc, savedir, type='rmse'):
    '''
    depth: (H,W,K,1)
    '''
    disp_mean = np.mean(disp_k,-2) # (H,W,1)
    if type=='rmse':
        disp_mse = (disp_mean-disp_gt)**2
    elif type=='mae':
        disp_mse = abs(disp_mean-disp_gt)
    mse_r = np.mean(disp_mse,-1).reshape(-1) # (N,)
    unc_r = disp_unc.reshape(-1)
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    mse_by_err, mse_by_unc = sparsification_plot(torch.tensor(unc_r), torch.tensor(mse_r), uncert_type='v', err_type=type)
    savedir_ause = os.path.join(savedir, 'disp_ause')
    os.makedirs(savedir_ause, exist_ok=True)
    with open(savedir_ause + '/{:s}_by_error_{:02d}.npy'.format(type,img_i),'wb') as f:
        np.save(f,mse_by_err)
    with open(savedir_ause + '/{:s}_by_std_{:02d}.npy'.format(type,img_i),'wb') as f:
        np.save(f,mse_by_unc)
    ause = np.trapz(mse_by_unc - mse_by_err, ratio_removed)
    ausc = np.trapz(mse_by_unc, ratio_removed)
    print('- AUSC metric std - {:s} is {:.5f}.'.format(type, ausc))
    print('- AUSE metric std - {:s} is {:.5f}.'.format(type, ause))
    plt.clf()
    plt.plot(ratio_removed, mse_by_err, '--')
    plt.plot(ratio_removed, mse_by_unc, '-r')
    plt.grid()
    plt.savefig(savedir_ause+'/{:s}_ause_{:02d}.png'.format(type,img_i))

    return ause, ausc