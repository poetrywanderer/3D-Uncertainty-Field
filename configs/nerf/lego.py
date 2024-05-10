_base_ = '../default.py'

expname = 'lego'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='../Data/nerf_synthetic/lego',
    dataname='lego',
    dataset_type='blender',
    white_bkgd=True,
)