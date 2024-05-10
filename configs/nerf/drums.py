_base_ = '../default.py'

expname = 'drums'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='../Data/nerf_synthetic/drums',
    dataname='drums',
    dataset_type='blender',
    white_bkgd=True,
)

