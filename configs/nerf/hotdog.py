_base_ = '../default.py'

expname = 'hotdog'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='../Data/nerf_synthetic/hotdog',
    dataname='hotdog',
    dataset_type='blender',
    white_bkgd=True,
)

