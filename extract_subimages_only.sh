#!/bin/bash

# 检查参数数量
if [ $# -ne 2 ]; then
    echo "用法: $0 <dataset_root> <dataset_name>"
    echo "示例: $0 /root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite aigc_ocr_part1_img"
    exit 1
fi

# 获取参数
dataset_root=$1
dataset_name=$2

# 检查数据集根目录是否存在
if [ ! -d "$dataset_root" ]; then
    echo "错误: 数据集根目录 '$dataset_root' 不存在"
    exit 1
fi

# 定义多尺度数据集路径
multiscale_path="$dataset_root/${dataset_name}_multiscale"
multiscale_sub_path="$dataset_root/${dataset_name}_multiscale_sub"

# 检查多尺度数据集是否存在
if [ ! -d "$multiscale_path" ]; then
    echo "错误: 多尺度数据集 '$multiscale_path' 不存在"
    echo "请确保已运行步骤1生成多尺度数据集，或检查路径是否正确"
    exit 1
fi

echo "开始提取子图像..."
echo "数据集根目录: $dataset_root"
echo "数据集名称: $dataset_name"
echo "多尺度数据集路径: $multiscale_path"
echo "子图像输出路径: $multiscale_sub_path"

# 创建多尺度子图像目录
echo "创建子图像输出目录: $multiscale_sub_path"
mkdir -p "$multiscale_sub_path"
if [ $? -ne 0 ]; then
    echo "错误: 无法创建子图像输出目录"
    exit 1
fi

# 执行步骤2: 提取子图像
echo "执行步骤2: 提取子图像..."
python scripts/extract_subimages.py --input "$multiscale_path" --output "$multiscale_sub_path" --crop_size 512 --step 256

# 检查是否成功
if [ $? -eq 0 ]; then
    echo "子图像提取完成!"
    echo "输出路径: $multiscale_sub_path"
else
    echo "错误: 子图像提取失败"
    exit 1
fi