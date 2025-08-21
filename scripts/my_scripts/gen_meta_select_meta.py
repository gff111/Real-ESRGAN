import os
import random

def sample_meta_lines_custom(original_meta_path, output_meta_path, samples_config):
    """
    从原始meta文件中按文件夹抽取指定数量的行，可以为不同文件夹设置不同的采样数量
    
    Args:
        original_meta_path: 原始meta文件路径
        output_meta_path: 输出meta文件路径
        samples_config: 字典，键为文件夹名，值为要抽取的样本数量
    """
    
    # 按文件夹组织数据
    folder_lines = {}
    
    # 读取原始meta文件
    with open(original_meta_path, 'r') as f:
        lines = f.readlines()
    
    # 按文件夹分组
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('/')
        if len(parts) > 1:
            folder_name = parts[0]
            if folder_name not in folder_lines:
                folder_lines[folder_name] = []
            folder_lines[folder_name].append(line)
    
    # 从每个文件夹中抽取指定数量的样本
    selected_lines = []
    for folder, lines_in_folder in folder_lines.items():
        samples_needed = samples_config.get(folder, 0)  # 默认抽取0个
        
        if samples_needed <= 0:
            print(f"文件夹 {folder}: 跳过（未配置采样数量）")
            continue
            
        if len(lines_in_folder) <= samples_needed:
            selected_lines.extend(lines_in_folder)
            print(f"文件夹 {folder}: 选择了所有 {len(lines_in_folder)} 个样本（需要 {samples_needed} 个）")
        else:
            sampled = random.sample(lines_in_folder, samples_needed)
            selected_lines.extend(sampled)
            print(f"文件夹 {folder}: 从 {len(lines_in_folder)} 个样本中随机选择了 {samples_needed} 个")
    
    # 打乱顺序
    random.shuffle(selected_lines)
    
    # 写入新的meta文件
    with open(output_meta_path, 'w') as f:
        for line in selected_lines:
            f.write(line + '\n')
    
    print(f"\n完成！共选择了 {len(selected_lines)} 个样本")
    print(f"新meta文件已保存到: {output_meta_path}")

# 使用示例
if __name__ == "__main__":
    # 根据你提供的路径配置采样数量
    samples_config = {
        "DIV2K_train_HR_multiscale_sub": 7000,
        "sa_text_high_multiscale_sub": 10000,
        "Flickr2K_HR_multiscale_sub": 22000,
        "tusou_es_clarity_gt_0_9_multiscale_sub": 200000,
        "aigc_ocr_part1_img_multiscale_sub": 50000,
        "aigc_ocr_gt_30_multiscale_sub": 50000,

        "52mm_multiscale_sub": 6500,
        "poster_text_images_high_multiscale_sub": 100000,
        "computer_sample_multiscale_sub": 13000,
        "east_report_sample_multiscale_sub": 13000,
        "map_multiscale_sub": 9200,
        "stocks_financial_reports_pdf_sample_multiscale_sub": 13000
    }
    
    # 替换为你的实际文件路径
    original_meta_path = "/root/paddlejob/workspace/env_run/guanfeiqiang/datasets/meta_info_12data_mutiscale_sub.txt"
    output_meta_path = "/root/paddlejob/workspace/env_run/guanfeiqiang/datasets/meta_info_v13_mutiscale_sub.txt"
    
    sample_meta_lines_custom(original_meta_path, output_meta_path, samples_config)