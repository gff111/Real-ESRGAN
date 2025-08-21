import os
from pathlib import Path
from collections import defaultdict

def count_images_in_folders(root_folder):
    """
    统计指定文件夹下所有子文件夹中的图片数量
    
    Args:
        root_folder (str): 要统计的根文件夹路径
    
    Returns:
        dict: 包含每个文件夹及其图片数量的字典
    """
    # 支持的图片文件扩展名
    image_extensions = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', 
        '.tiff', '.tif', '.webp', '.svg', '.raw',
        '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP'
    }
    
    # 用于存储统计结果
    folder_stats = defaultdict(int)
    total_images = 0
    
    # 检查文件夹是否存在
    if not os.path.exists(root_folder):
        print(f"错误：文件夹 '{root_folder}' 不存在")
        return {}
    
    if not os.path.isdir(root_folder):
        print(f"错误：'{root_folder}' 不是一个文件夹")
        return {}
    
    print(f"开始统计文件夹: {root_folder}")
    print("-" * 50)
    
    # 遍历所有子文件夹
    for foldername, subfolders, filenames in os.walk(root_folder):
        image_count = 0
        
        # 统计当前文件夹中的图片数量
        for filename in filenames:
            file_extension = os.path.splitext(filename)[1]
            if file_extension in image_extensions:
                image_count += 1
        
        # 只记录包含图片的文件夹
        if image_count > 0:
            folder_path = Path(foldername)
            # 获取相对于根文件夹的路径
            relative_path = folder_path.relative_to(root_folder) if folder_path != Path(root_folder) else "当前文件夹"
            folder_stats[str(relative_path)] = image_count
            total_images += image_count
    
    return folder_stats, total_images

def print_statistics(folder_stats, total_images):
    """
    打印统计结果
    
    Args:
        folder_stats (dict): 文件夹统计信息
        total_images (int): 图片总数
    """
    if not folder_stats:
        print("没有找到任何图片文件")
        return
    
    # 按图片数量排序
    sorted_folders = sorted(folder_stats.items(), key=lambda x: x[1], reverse=True)
    
    print("文件夹图片数量统计结果：")
    print("=" * 50)
    
    for folder, count in sorted_folders:
        print(f"{folder}: {count} 张图片")
    
    print("=" * 50)
    print(f"总计: {len(folder_stats)} 个文件夹, {total_images} 张图片")

def save_to_file(folder_stats, total_images, output_file="image_statistics.txt"):
    """
    将统计结果保存到文件
    
    Args:
        folder_stats (dict): 文件夹统计信息
        total_images (int): 图片总数
        output_file (str): 输出文件名
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("文件夹图片数量统计结果\n")
        f.write("=" * 50 + "\n")
        
        sorted_folders = sorted(folder_stats.items(), key=lambda x: x[1], reverse=True)
        
        for folder, count in sorted_folders:
            f.write(f"{folder}: {count} 张图片\n")
        
        f.write("=" * 50 + "\n")
        f.write(f"总计: {len(folder_stats)} 个文件夹, {total_images} 张图片\n")
    
    print(f"\n统计结果已保存到: {output_file}")

if __name__ == "__main__":
    # 用户输入文件夹路径
    folder_path = '/root/paddlejob/workspace/env_run/guanfeiqiang/datasets/dataset_lite'
    
    # 如果输入为空，使用当前文件夹
    if not folder_path:
        folder_path = "."
    
    # 统计图片数量
    folder_stats, total_images = count_images_in_folders(folder_path)
    
    # 打印结果
    print_statistics(folder_stats, total_images)
    
    # # 询问是否保存结果到文件
    # save_option = input("\n是否将结果保存到文件？(y/n): ").strip().lower()
    # if save_option in ['y', 'yes', '是']:
    #     output_filename = input("请输入输出文件名 (默认为 image_statistics.txt): ").strip()
    #     if not output_filename:
    #         output_filename = "image_statistics.txt"
    #     save_to_file(folder_stats, total_images, output_filename)