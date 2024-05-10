import numpy as np

from .load_llff import load_llff_data
from .load_blender import load_blender_data_test

def load_data(args):

    K, depths = None, None
    near_clip = None

    if args.dataset_type == 'llff':
        images, depths, poses, bds, render_poses, sc = load_llff_data(
                args.datadir, args.factor, args.width, args.height,
                recenter=True, bd_factor=args.bd_factor,
                spherify=args.spherify,
                load_depths=args.load_depths,
                movie_render_kwargs=args.movie_render_kwargs)
        
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        i_all = np.arange(images.shape[0])

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test)])

        if args.dataname == 'outdoor1':
            i_train_range = list(np.arange(19,61)) + list(np.arange(101,131)) + list(np.arange(155,179)) + list(np.arange(195,210))
            i_train = [i for i in i_train_range if i not in i_test]

        if args.dataname == 'outdoor2':
            i_train_range = list(np.arange(0,48)) + list(np.arange(96,136)) + list(np.arange(168,206)) + list(np.arange(236,264))
            i_train = [i for i in i_train_range if i not in i_test]
         
        i_val = i_test

        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            # near_clip = max(np.ndarray.min(bds) * .9, 0)
            near_clip = max(np.ndarray.min(bds), 0) if np.ndarray.min(bds) > 0 else 0.6
            _far = max(np.ndarray.max(bds) * 1., 0)
            near = 0
            far = inward_nearfar_heuristic(poses[i_train, :3, 3])[1]
            print('near_clip', near_clip)
            print('original far', _far)
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_train, i_test = load_blender_data_test(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_all = np.arange(images.shape[0])

        if args.dataname == 'lego':
            i_train_range = list(np.arange(40,60)) + list(np.arange(140,160))
        
        i_train = [i for i in i_train_range if i not in i_test]

        i_val = i_test

        near, far = 2., 6.

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_all=i_all, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
    )

    if args.dataset_type == 'llff': data_dict.update(sc=sc)
        
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far

