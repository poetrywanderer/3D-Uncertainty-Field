# 3D-Uncertainty-Field
ICRA 2024 -- Estimating 3D Uncertainty Field: Quantifying Uncertainty for Neural Radiance Fields

[Paper](https://arxiv.org/abs/2311.01815), [Video]()

### Results: 2D Uncertainty

<img src="https://github.com/poetrywanderer/3D-Uncertainty-Field/blob/main/assets/2d_uncertainty.png" width="100%">

### Features

- A poc-hoc framwork to efficiently estimate 3D Uncertainty Field for NeRF-based methods. 

- Supported datasets:
    - *Bounded inward-facing*: [NeRF-Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
    - *Unbounded inward-facing*: [Real-Occluded](https://drive.google.com/drive/folders/1jGYoA2JwNYAo0vLrKa_0DZ9KiUy4PRKB?usp=sharing)

### Installation
```
git clone https://github.com/poetrywanderer/3D-Uncertainty-Field.git
pip install -r requirements.txt
```
Choose proper [Pytorch](https://pytorch.org/) version for your machine. [torch_scatter](https://github.com/rusty1s/pytorch_scatter) installation is needed following the framework of [DVGO](https://github.com/sunset1995/DirectVoxGO). 

<details>
  <summary> Dependencies (click to expand) </summary>

  - `PyTorch`, `numpy`, `torch_scatter`: main computation.
  - `scipy`, `lpips`: SSIM and LPIPS evaluation.
  - `tqdm`: progress bar.
  - `mmengine`: config system. Note that `mmcv` is not valid for recent enrionments.
  - `opencv-python`: image processing.
  - `imageio`, `imageio-ffmpeg`: images and videos I/O.
  - `Ninja`: to build the newly implemented torch extention just-in-time.
  - `einops`: torch tensor shaping with pretty api.
  - `torch_efficient_distloss`: O(N) realization for the distortion loss.
</details>


## Datasets Structure

The folder for datasets should be like:
<details>
  <summary> (click to expand;) </summary>

    Data
    ├── nerf_synthetic     # Link: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
    │   └── [chair|drums|hotdog|lego|materials]
    │       ├── [train|val|test]
    │       │   └── r_*.png
    │       └── transforms_[train|val|test].json
    │
    ├── Real-Occluded             # Link: 
    │   └── [indoor1|indoor2|outdoor1|outdoor2]
    │       ├── poses_bounds.npy
    │       └── [images_2|images_4]
    │
</details>


## Running

```bash
$ bash train.sh
```
<details>
  <summary> Command Line Arguments for run.py </summary>

  - --config: the scene configs, eg. `configs/nerf/lego.py`
  - --render_test: render test set
  - --render_train: render train set
  - --eval_ssim: evaluate metric SSIM
  - --eval_ause: evaluate metric ause
  - --eval_lpips_alex: evaluate metric LPIPS
  - --eval_lpips_vgg: evaluate metric LPIPS
  - --update_uncertainty_fine: update the estimated uncertainty field after training
  - --export_bbox_and_cams_only: Used to inspect the camera and the allocated BBox
  - --export_coarse_only: Used to inspect the learned geometry after coarse optimization.
  - --export_fine_only: Used to inspect the learned geometry after fine optimization.

</details>

### Visualization

We provide two jupyter scripts to visualize the results:
- vis_points_bounded.ipynb
- vis_points_unbounded.ipynb

## Related works
- **S-NeRF** — [Stochastic Neural Radiance Fields: Quantifying Uncertainty in Implicit 3D Representations, 3DV 2021](https://arxiv.org/abs/2109.02123). The first to incorporate uncertainty estimation into NeRF, from the best of our knowledge.
- **CF-NeRF** — [Conditional-Flow NeRF: Accurate 3D Modelling with Reliable Uncertainty Quantification, ECCV 2022](https://arxiv.org/abs/2203.10192). A more compressive and flexible framwork for more complex scenes, compared to S-NeRF. 
- **Bayes' Rays** — [Bayes' Rays: Uncertainty Quantification for Neural Radiance Fields, CVPR 2024](https://bayesrays.github.io/). An awesome post-hoc framework to establish a volumetric uncertainty field using spatial perturbations and a Bayesian Laplace approximation. 

## Acknowledgement

The code framework is origined from the awesome [DVGO](https://github.com/sunset1995/DirectVoxGO) implementation. 
