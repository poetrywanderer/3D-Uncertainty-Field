_base_ = '../default.py'

expname = 'chair'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='../Data/nerf_synthetic/chair',
    dataname='chair',
    dataset_type='blender',
    white_bkgd=True,
)

