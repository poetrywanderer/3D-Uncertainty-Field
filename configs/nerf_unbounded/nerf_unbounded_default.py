_base_ = '../default.py'

basedir = './logs/nerf_unbounded'

data = dict(
    dataset_type='llff',
    datatype='unbounded',
    spherify=True,
    factor=8,
    llffhold=8,
    white_bkgd=True,
    rand_bkgd=True,
    unbounded_inward=True,
    load2gpu_on_the_fly=True,
)

coarse_train = dict(N_iters=0)

# used for dense inputs
fine_train = dict(
    N_iters=5000,
    N_rand=2048,
    lrate_decay=80,
    uncertainty=True,
    ray_sampler='flatten',
    weight_distortion=0.01,
    weight_nearclip=1.0,
    lrate_k0=0.01,
    weight_density=0.001,
    pg_scale=[1000,2000,3000,4000], # note: too many is harmful
    tv_before=12000,
    tv_dense_before=12000,
    weight_tv_density=1e-5, # 1e-3 is too big for training all views, causing oversmooth geometry, particularly background
    weight_tv_k0=1e-6,
)

alpha_init = 1e-4
stepsize = 0.5

fine_model_and_render = dict(
    num_voxels=201**3,
    num_voxels_base=201**3,
    alpha_init=alpha_init,
    stepsize=stepsize,
    uncertainty_mask=False,
    density_std_init=1,
    K_samples=12,
    bg_len=0.2,
    glob_lambda_den=1,
    fast_color_thres={
        '_delete_': True,
        0   : alpha_init*stepsize/10,
        1500: min(alpha_init, 1e-4)*stepsize/5,
        2500: min(alpha_init, 1e-4)*stepsize/2,
        3500: min(alpha_init, 1e-4)*stepsize/1.5,
        4500: min(alpha_init, 1e-4)*stepsize,
        5500: min(alpha_init, 1e-4),
        6500: 1e-4,
    },
    world_bound_scale=1,
)