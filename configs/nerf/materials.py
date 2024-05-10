_base_ = '../default.py'

expname = 'materials'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='../Data/nerf_synthetic/materials',
    dataname='materials',
    dataset_type='blender',
    white_bkgd=True,
)

