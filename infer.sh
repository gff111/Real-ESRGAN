# x2 sr，一定要指定-s为2，不然输出仍是x4
# python inference_realesrgan.py -n RealESRGAN_x2plus -i inputs --output ./results/face_enhance_x2_r -s 2
# python inference_realesrgan.py -n RealESRGAN_x2plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face -s 2
# x4 sr
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_imgs --output ./results/face_enhance_x4_r
# x4 sr with face restoration
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_imgs --output ./results/x4_with_face_enhance_x4 --face_enhance


# baidu model x4
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_imgs --output ./results/general_x4_baidu --model_path weights/BAIDU_RealESRGAN_x4plus_v3.pth


# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/test_set_sr_baidu_tile_512 --model_path weights/BAIDU_RealESRGAN_x4plus_v3.pth --tile 512
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/origin_esrgan --model_path ./weights/RealESRGAN_x4plus.pth --tile 512
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/test_set_sr --output ./results/origin_esrnet --model_path ./weights/RealESRNet_x4plus.pth --tile 512

# python inference_realesrgan.py -n RealESRGAN_x2plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face_enhance -s 2 --face_enhance
# python inference_realesrgan.py -n RealESRGAN_x2plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face_enhance -s 2 --face_enhance
# python inference_realesrgan.py -n RealESRGAN_x2plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face_enhance -s 2 --face_enhance
# python inference_realesrgan.py -n RealESRGAN_x1plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face_x1_890000G_16bits -s 1 --out_16bit

# python inference_realesrgan.py -n RealESRGAN_x1plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face_x1_gan_400000G -s 1
# checkpoint=("5000" "10000" "20000" "50000" "100000" "150000" "200000" "250000" "300000" "350000")
# for item in "${checkpoint[@]}"
# do
# ori_weight=/data/yh/SR2023/Real-ESRGAN/experiments/train_RealESRGANx2plus_400k_B12G4/models/net_g_${item}.pth
# target=./weights/RealESRGAN_x1plus.pth
# target_bk=./weights/RealESRGAN_x1plus_gan_${item}G.pth
# ls $ori_weight
# echo $target
# echo $target_bk
# cp -r $ori_weight $target
# python inference_realesrgan.py -n RealESRGAN_x1plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face_x1_gan_${item}G -s 1
# mv $target $target_bk
# done