#!/bin/bash

# 生成第一个元数据文件
# python scripts/generate_meta_info.py \
#     --input \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/aigc_ocr_part1_img_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/Flickr2K_HR_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/DIV2K_train_HR_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/52mm_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/sa_text_high_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/poster_text_images_high_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/tusou_es_clarity_gt_0_9_multiscale_sub \
#     --root \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#     --meta_info /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/meta_info_df2k_aigcocr_realce_satext_poster_tusou.txt

# 生成第二个元数据文件 (包含更多数据集)
# python scripts/generate_meta_info.py \
#     --input \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/aigc_ocr_part1_img_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/Flickr2K_HR_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/DIV2K_train_HR_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/52mm_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/sa_text_high_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/poster_text_images_high_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/tusou_es_clarity_gt_0_9_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/map_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/computer_sample_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/east_report_sample_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/stocks_financial_reports_pdf_sample_multiscale_sub \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/aigc_ocr_gt_30_multiscale_sub \
#     --root \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#         /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
#     --meta_info /root/paddlejob/workspace/env_run/zhuyinghao/datasets/meta_info_v20.txt


# 输出文字
python scripts/generate_meta_info.py \
    --input \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/52mm_multiscale_sub \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/sa_text_high_multiscale_sub \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/poster_text_images_high_multiscale_sub \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/map_multiscale_sub \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/computer_sample_multiscale_sub \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/east_report_sample_multiscale_sub \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/stocks_financial_reports_pdf_sample_multiscale_sub \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/aigc_ocr_gt_30_multiscale_sub \
    --root \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
        /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite \
    --meta_info /root/paddlejob/workspace/env_run/zhuyinghao/datasets/meta_info_v30_word.txt