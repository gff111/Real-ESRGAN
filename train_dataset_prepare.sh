dataset_root=/root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite

python scripts/generate_multiscale_DF2K.py --input $dataset_root/aigc_ocr_part1_img --output $dataset_root/aigc_ocr_part1_img_multiscale
python scripts/extract_subimages.py --input $dataset_root/aigc_ocr_part1_img_multiscale --output $dataset_root/aigc_ocr_part1_img_multiscale_sub --crop_size 512 --step 256

python scripts/generate_multiscale_DF2K.py --input $dataset_root/tusou_es_clarity_gt_0_9 --output $dataset_root/tusou_es_clarity_gt_0_9_multiscale
python scripts/extract_subimages.py --input $dataset_root/tusou_es_clarity_gt_0_9_multiscale --output $dataset_root/tusou_es_clarity_gt_0_9_multiscale_sub --crop_size 512 --step 256

# python scripts/extract_subimages.py --input /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset/hr --output /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset/hr_sub --crop_size 400 --step 200
# python scripts/generate_meta_info.py --input /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset/hr_sub /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset/hr_multiscale_sub --root /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset --meta_info /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset/meta_info_DF2K_OSTmultiscale_sub_and_DF2K_OST_sub.txt


# python scripts/generate_meta_info.py --input /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset/hr_sub /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset/hr_multiscale_sub --root /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset --meta_info /mnt/ec-data2/ivs/1080p/zyh/df2k_ost_dataset/meta_info_DF2K_OSTmultiscale_sub_and_DF2K_OST_sub.txt