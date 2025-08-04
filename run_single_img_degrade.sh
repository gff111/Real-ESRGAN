python degrade_single_image.py \
    --config options/finetune_realesrgan_x4plus.yml \
    --input /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/aigc_ocr_part1_img_multiscale_sub/cndroop_location_d1a5885efa3d35a9f634575e9a0cab1cT1_s003.png \
    --output ./degrade.png \
    --device cuda \
    --seed 42