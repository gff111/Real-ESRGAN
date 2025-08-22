checkpoint=("120000" "170000" "210000" "220000")
for item in "${checkpoint[@]}"
do
ori_weight=/root/paddlejob/workspace/env_run/guanfeiqiang/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_v13_no_noise_deDeg_upLr/models/net_g_${item}.pth
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output results/gfq/gfq_v13_${item}G --model_path $ori_weight --tile 512
done