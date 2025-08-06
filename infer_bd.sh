# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/test_set_sr_baidu_tile_512 --model_path weights/BAIDU_RealESRGAN_x4plus_v3.pth --tile 512
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/test_set_sr_origin_tile_512 --tile 512


# checkpoint=("5000" "10000" "20000" "50000" "100000" "150000" "200000" "250000" "300000" "350000")
# for item in "${checkpoint[@]}"
# do
# ori_weight=/root/paddlejob/workspace/qiucan/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_qc_text725/models/net_g_${item}.pth
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/qiucan40_${item}G --model_path $ori_weight --tile 512
# done

# checkpoint=("5000" "10000" "20000" "50000" "60000" "70000" "80000" "90000")
# for item in "${checkpoint[@]}"
# do
# ori_weight=/root/paddlejob/workspace/env_run/zhuyinghao/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_yh/models/net_g_${item}.pth
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/yinghao10/yinghao10_${item}G --model_path $ori_weight --tile 512
# done


# # yinghao20
# checkpoint=("5000" "10000" "20000" "25000")
# for item in "${checkpoint[@]}"
# do
# ori_weight=/root/paddlejob/workspace/env_run/zhuyinghao/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_yh_no_usm/models/net_g_${item}.pth
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/yinghao20/yinghao20_${item}G --model_path $ori_weight --tile 512
# done

# # yinghao30

# checkpoint=("5000" "10000" "20000")
# for item in "${checkpoint[@]}"
# do
# ori_weight=/root/paddlejob/workspace/env_run/zhuyinghao/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_yh_30_online_degrade/models/net_g_${item}.pth
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/yinghao30/yinghao30_${item}G --model_path $ori_weight --tile 512
# done

# yinghao40
# checkpoint=("5000" "10000" "20000" "30000")
# for item in "${checkpoint[@]}"
# do
# ori_weight=/root/paddlejob/workspace/env_run/zhuyinghao/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_yh_30_online_degrade_jpeg2_10_70/models/net_g_${item}.pth
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/yinghao40/yinghao40_${item}G --model_path $ori_weight --tile 512
# done

# yinghao50
checkpoint=("5000" "10000" "15000")
for item in "${checkpoint[@]}"
do
ori_weight=/root/paddlejob/workspace/env_run/zhuyinghao/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_yh_30_online_degrade_jpeg2_10_70_x2/models/net_g_${item}.pth
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/yinghao50/yinghao50_${item}G --model_path $ori_weight --tile 512
done
