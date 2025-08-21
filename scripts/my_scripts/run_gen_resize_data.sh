#!/bin/bash

# 定义输入数据集路径数组
input_datasets=(
    "/root/paddlejob/workspace/env_run/rmp-individual/guanfeiqiang/datasets/sr_data/DIV2K_train_HR"
    "/root/paddlejob/workspace/env_run/rmp-individual/guanfeiqiang/datasets/sr_data/Flickr2K_HR"
    "/root/paddlejob/workspace/env_run/rmp-individual/guanfeiqiang/datasets/sr_data/RealCE"
    "/root/paddlejob/workspace/env_run/rmp-individual/guanfeiqiang/datasets/sr_data/sa_text"
    "/root/paddlejob/workspace/env_run/guanfeiqiang/datasets/raw_sr_data/aigc_ocr_gt_30"
    "/root/paddlejob/workspace/env_run/rmp-individual/zhuyinghao/sr_dataset_2025/aigc_ocr_part1_img"
    "/root/paddlejob/workspace/env_run/rmp-individual/zhuyinghao/sr_dataset_2025/tusou_es/tusou_es_clarity_gt_0_9"
    "/root/paddlejob/workspace/env_run/guanfeiqiang/datasets/raw_sr_data/doc_data/map"
    "/root/paddlejob/workspace/env_run/guanfeiqiang/datasets/raw_sr_data/doc_data/doc_img_sample/computer_sample"
    "/root/paddlejob/workspace/env_run/guanfeiqiang/datasets/raw_sr_data/doc_data/doc_img_sample/east_report_sample"
    "/root/paddlejob/workspace/env_run/guanfeiqiang/datasets/raw_sr_data/doc_data/doc_img_sample/stocks_financial_reports_pdf_sample"
    "/root/paddlejob/workspace/env_run/rmp-individual/qiucan/datasets/poster_text_images_high"
)

# 定义输出根路径
output_root="/root/paddlejob/workspace/env_run/guanfeiqiang/datasets/dataset_lite_sub_only"

# 创建输出根目录
echo "创建输出根目录: $output_root"
mkdir -p "$output_root"
if [ $? -ne 0 ]; then
    echo "错误: 无法创建输出根目录"
    exit 1
fi

# 遍历所有数据集进行处理
for input_path in "${input_datasets[@]}"; do
    # 检查输入数据集是否存在
    if [ ! -d "$input_path" ]; then
        echo "警告: 数据集 '$input_path' 不存在，跳过"
        continue
    fi

    # 从完整路径中提取数据集名称
    dataset_name=$(basename "$input_path")
    
    # 对于嵌套路径，使用更具体的名称
    if [[ "$input_path" == *"tusou_es_clarity_gt_0_9"* ]]; then
        dataset_name="tusou_es_clarity_gt_0_9"
    elif [[ "$input_path" == *"computer_sample"* ]]; then
        dataset_name="doc_computer_sample"
    elif [[ "$input_path" == *"east_report_sample"* ]]; then
        dataset_name="doc_east_report_sample"
    elif [[ "$input_path" == *"stocks_financial_reports_pdf_sample"* ]]; then
        dataset_name="doc_stocks_financial_reports_sample"
    fi

    echo "=============================================="
    echo "开始处理数据集: $dataset_name"
    echo "输入路径: $input_path"

    # 定义子图像输出路径
    sub_only_path="$output_root/${dataset_name}_sub_only"

    echo "子图像输出路径: $sub_only_path"

    # 创建子图像输出目录
    echo "创建子图像输出目录: $sub_only_path"
    mkdir -p "$sub_only_path"
    if [ $? -ne 0 ]; then
        echo "错误: 无法创建子图像输出目录"
        continue
    fi

    # 提取子图像
    echo "步骤: 提取子图像..."
    python /root/paddlejob/workspace/env_run/guanfeiqiang/Real-ESRGAN/scripts/extract_subimages.py --input "$input_path" --output "$sub_only_path" --crop_size 512 --step 256

    # 检查是否成功
    if [ $? -ne 0 ]; then
        echo "错误: 提取子图像失败"
        continue
    fi

    echo "数据集 '$dataset_name' 处理完成!"
    echo "生成的子图像保存在: $sub_only_path"
    echo "=============================================="
    echo ""
done

echo "所有数据集处理完成!"
echo "总输出目录: $output_root"