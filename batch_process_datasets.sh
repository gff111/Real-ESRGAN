#!/bin/bash

# 设置数据集根目录
dataset_root="/root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite"

# 定义要处理的数据集名称列表
datasets=("aigc_ocr_part1_img" "tusou_es_clarity_gt_0_9" "Flickr2K_HR" "DIV2K_train_HR" "52mm" "sa_text_high")

echo "开始批量处理数据集..."
echo "数据集根目录: $dataset_root"
echo "要处理的数据集: ${datasets[@]}"
echo "=================================="

# 遍历每个数据集进行处理
for dataset_name in "${datasets[@]}"; do
    echo ""
    echo "正在处理数据集: $dataset_name"
    echo "----------------------------------"

    # 调用 train_dataset_prepare.sh 脚本
    ./train_dataset_prepare.sh "$dataset_root" "$dataset_name"

    # 检查处理结果
    if [ $? -eq 0 ]; then
        echo "数据集 '$dataset_name' 处理成功!"
    else
        echo "错误: 数据集 '$dataset_name' 处理失败!"
        echo "是否继续处理下一个数据集? (y/n)"
        read -r continue_processing
        if [[ ! "$continue_processing" =~ ^[Yy]$ ]]; then
            echo "用户选择停止处理，退出脚本"
            exit 1
        fi
    fi

    echo "----------------------------------"
done

echo ""
echo "所有数据集处理完成!"
echo "处理的数据集: ${datasets[@]}"