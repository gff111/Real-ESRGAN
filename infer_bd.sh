# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/test_set_sr_baidu_tile_512 --model_path weights/BAIDU_RealESRGAN_x4plus_v3.pth --tile 512
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/test_set_sr_origin_tile_512 --tile 512


# checkpoint=("5000" "10000" "20000" "50000" "100000" "150000" "200000" "250000" "300000" "350000")
# for item in "${checkpoint[@]}"
# do
# ori_weight=/root/paddlejob/workspace/qiucan/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_qc_text725/models/net_g_${item}.pth
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/qiucan40_${item}G --model_path $ori_weight --tile 512
# done

checkpoint=("5000" "10000" "20000" "50000")
for item in "${checkpoint[@]}"
do
ori_weight=/root/paddlejob/workspace/qiucan/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_400k_qc_text2/models/net_g_${item}.pth
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/qiucan30/qiucan30_${item}G --model_path $ori_weight --tile 512
done