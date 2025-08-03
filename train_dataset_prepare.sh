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

# 检查原始数据集是否存在
original_dataset_path="$dataset_root/$dataset_name"
if [ ! -d "$original_dataset_path" ]; then
    echo "错误: 数据集 '$original_dataset_path' 不存在"
    exit 1
fi

echo "开始处理数据集: $dataset_name"
echo "数据集根目录: $dataset_root"
echo "原始数据集路径: $original_dataset_path"

# 定义中间路径
multiscale_path="$dataset_root/${dataset_name}_multiscale"
multiscale_sub_path="$dataset_root/${dataset_name}_multiscale_sub"

echo "多尺度数据集路径: $multiscale_path"
echo "多尺度子图像路径: $multiscale_sub_path"

# 创建多尺度数据集目录
echo "创建多尺度数据集目录: $multiscale_path"
mkdir -p "$multiscale_path"
if [ $? -ne 0 ]; then
    echo "错误: 无法创建多尺度数据集目录"
    exit 1
fi

# 创建多尺度子图像目录
echo "创建多尺度子图像目录: $multiscale_sub_path"
mkdir -p "$multiscale_sub_path"
if [ $? -ne 0 ]; then
    echo "错误: 无法创建多尺度子图像目录"
    exit 1
fi

# 步骤1: 生成多尺度数据集
echo "步骤1: 生成多尺度数据集..."
python scripts/generate_multiscale_DF2K.py --input "$original_dataset_path" --output "$multiscale_path" --n_thread 16

# 检查步骤1是否成功
if [ $? -ne 0 ]; then
    echo "错误: 生成多尺度数据集失败"
    exit 1
fi

# 步骤2: 提取子图像
echo "步骤2: 提取子图像..."
python scripts/extract_subimages.py --input "$multiscale_path" --output "$multiscale_sub_path" --crop_size 512 --step 256

# 检查步骤2是否成功
if [ $? -ne 0 ]; then
    echo "错误: 提取子图像失败"
    exit 1
fi

echo "数据集 '$dataset_name' 处理完成!"
echo "生成的文件:"
echo "  - 多尺度数据集: $multiscale_path"
echo "  - 多尺度子图像: $multiscale_sub_path"