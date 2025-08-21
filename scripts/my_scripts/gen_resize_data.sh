#!/bin/bash
##

# 检查参数数量
if [ $# -ne 3 ]; then
    echo "用法: $0 <dataset_root> <dataset_name> <output_root>"
    echo "示例: $0 /root/datasets/input aigc_ocr_part1_img /root/datasets/output"
    exit 1
fi

# 获取参数
dataset_root=$1
dataset_name=$2
output_root=$3

# 检查数据集根目录是否存在
if [ ! -d "$dataset_root" ]; then
    echo "错误: 数据集根目录 '$dataset_root' 不存在"
    exit 1
fi

# 检查输出根目录是否存在，如果不存在则创建
if [ ! -d "$output_root" ]; then
    echo "创建输出根目录: $output_root"
    mkdir -p "$output_root"
    if [ $? -ne 0 ]; then
        echo "错误: 无法创建输出根目录"
        exit 1
    fi
fi

# 检查原始数据集是否存在
original_dataset_path="$dataset_root/$dataset_name"
if [ ! -d "$original_dataset_path" ]; then
    echo "错误: 数据集 '$original_dataset_path' 不存在"
    exit 1
fi

echo "开始处理数据集: $dataset_name"
echo "数据集根目录: $dataset_root"
echo "输出根目录: $output_root"
echo "原始数据集路径: $original_dataset_path"

# 定义子图像输出路径
sub_only_path="$output_root/${dataset_name}_sub_only"

echo "子图像输出路径: $sub_only_path"

# 创建子图像输出目录
echo "创建子图像输出目录: $sub_only_path"
mkdir -p "$sub_only_path"
if [ $? -ne 0 ]; then
    echo "错误: 无法创建子图像输出目录"
    exit 1
fi

# 直接提取子图像
echo "步骤: 提取子图像..."
python scripts/extract_subimages.py --input "$original_dataset_path" --output "$sub_only_path" --crop_size 512 --step 256

# 检查是否成功
if [ $? -ne 0 ]; then
    echo "错误: 提取子图像失败"
    exit 1
fi

echo "数据集 '$dataset_name' 处理完成!"
echo "生成的子图像保存在: $sub_only_path"