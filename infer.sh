# x2 sr，一定要指定-s为2，不然输出仍是x4
# python inference_realesrgan.py -n RealESRGAN_x2plus -i inputs --output ./results/face_enhance_x2_r -s 2
# python inference_realesrgan.py -n RealESRGAN_x2plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face -s 2
# x4 sr
# python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --output ./results/face_enhance_x4_r

# python inference_realesrgan.py -n RealESRGAN_x2plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face_enhance -s 2 --face_enhance
# python inference_realesrgan.py -n RealESRGAN_x2plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face_enhance -s 2 --face_enhance
# python inference_realesrgan.py -n RealESRGAN_x2plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face_enhance -s 2 --face_enhance
python inference_realesrgan.py -n RealESRGAN_x1plus -i /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/png/select --output ./results/dirty_face_x1_890000G -s 1