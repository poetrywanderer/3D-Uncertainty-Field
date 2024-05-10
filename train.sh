# export CUDA_VISIBLE_DEVICES=0
# export TORCH_CUDA_ARCH_LIST="7.5+PTX"

python run.py --config configs/nerf/lego.py --export_bbox_and_cams_only --export_coarse_only --export_fine_only --render_test --update_uncertainty_fine
# python run.py --config configs/nerf_unbounded/outdoor2.py --update_uncertainty_fine --export_bbox_and_cams_only --export_fine_only --render_test #--render_train #--eval_ause --eval_lpips_alex --eval_ssim