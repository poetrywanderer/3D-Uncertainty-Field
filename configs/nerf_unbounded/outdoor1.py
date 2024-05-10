_base_ = './nerf_unbounded_default.py'

expname = 'outdoor1'

data = dict(
    datadir='../Data/outdoor1',
    dataname = 'outdoor1',
    factor=4, # 957x538
    movie_render_kwargs=dict(
        shift_x=0.0,  # positive right
        shift_y=-0.0, # negative down
        shift_z=0,
        scale_r=0.9,
        pitch_deg=-30,
    ),
)