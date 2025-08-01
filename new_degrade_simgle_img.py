import argparse
import cv2
import math
import numpy as np
import os
import random
import torch
import yaml
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.img_util import img2tensor, tensor2img
from torch.nn import functional as F


def load_config(config_path):
    """加载训练配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_kernels(config, kernel_range=None):
    """生成退化核"""
    if kernel_range is None:
        kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21

    # 第一个退化过程的核
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < config['datasets']['train']['sinc_prob']:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            config['datasets']['train']['kernel_list'],
            config['datasets']['train']['kernel_prob'],
            kernel_size,
            config['datasets']['train']['blur_sigma'],
            config['datasets']['train']['blur_sigma'],
            [-math.pi, math.pi],
            config['datasets']['train']['betag_range'],
            config['datasets']['train']['betap_range'],
            noise_range=None)

    # pad kernel to 21x21
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # 第二个退化过程的核
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < config['datasets']['train']['sinc_prob2']:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            config['datasets']['train']['kernel_list2'],
            config['datasets']['train']['kernel_prob2'],
            kernel_size,
            config['datasets']['train']['blur_sigma2'],
            config['datasets']['train']['blur_sigma2'],
            [-math.pi, math.pi],
            config['datasets']['train']['betag_range2'],
            config['datasets']['train']['betap_range2'],
            noise_range=None)

    # pad kernel2 to 21x21
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

    # 最终的sinc核
    if np.random.uniform() < config['datasets']['train']['final_sinc_prob']:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = torch.zeros(21, 21).float()
        sinc_kernel[10, 10] = 1

    return torch.FloatTensor(kernel), torch.FloatTensor(kernel2), sinc_kernel


def apply_degradation(img, config, kernels, device='cuda'):
    """对图像应用退化过程"""
    # 初始化工具
    jpeger = DiffJPEG(differentiable=False).to(device)
    usm_sharpener = USMSharp().to(device)

    # 转换为tensor并移到设备
    img_tensor = img2tensor([img], bgr2rgb=True, float32=True)[0].unsqueeze(0).to(device)
    kernel1, kernel2, sinc_kernel = [k.to(device) for k in kernels]

    # USM锐化
    img_usm = usm_sharpener(img_tensor)

    ori_h, ori_w = img_tensor.size()[2:4]

    # ----------------------- 第一个退化过程 ----------------------- #
    # 模糊
    out = filter2D(img_usm, kernel1)

    # 随机缩放
    updown_type = random.choices(['up', 'down', 'keep'], config['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, config['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(config['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)

    # 添加噪声
    gray_noise_prob = config['gray_noise_prob']
    if np.random.uniform() < config['gaussian_noise_prob']:
        from basicsr.data.degradations import random_add_gaussian_noise_pt
        out = random_add_gaussian_noise_pt(
            out, sigma_range=config['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        from basicsr.data.degradations import random_add_poisson_noise_pt
        out = random_add_poisson_noise_pt(
            out,
            scale_range=config['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)

    # JPEG压缩
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*config['jpeg_range'])
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- 第二个退化过程 ----------------------- #
    # 模糊
    if np.random.uniform() < config['second_blur_prob']:
        out = filter2D(out, kernel2)

    # 随机缩放
    updown_type = random.choices(['up', 'down', 'keep'], config['resize_prob2'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, config['resize_range2'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(config['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h / config['scale'] * scale), int(ori_w / config['scale'] * scale)), mode=mode)

    # 添加噪声
    gray_noise_prob = config['gray_noise_prob2']
    if np.random.uniform() < config['gaussian_noise_prob2']:
        from basicsr.data.degradations import random_add_gaussian_noise_pt
        out = random_add_gaussian_noise_pt(
            out, sigma_range=config['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        from basicsr.data.degradations import random_add_poisson_noise_pt
        out = random_add_poisson_noise_pt(
            out,
            scale_range=config['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)

    # JPEG压缩 + 最终的sinc滤波器
    if np.random.uniform() < 0.5:
        # resize back + 最终的sinc滤波器
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // config['scale'], ori_w // config['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)
        # JPEG压缩
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*config['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG压缩
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*config['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + 最终的sinc滤波器
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // config['scale'], ori_w // config['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)

    # 裁剪和四舍五入
    out = torch.clamp((out * 255.0).round(), 0, 255) / 255.

    return out.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description='对单张图像应用Real-ESRGAN退化过程')
    parser.add_argument('--config', type=str, required=True, help='训练配置文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, required=True, help='输出图像路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # 加载配置
    config = load_config(args.config)

    # 读取图像
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {args.input}")

    # 数据增强
    img = augment(img, config['datasets']['train']['use_hflip'], config['datasets']['train']['use_rot'])

    # 裁剪或填充到指定大小
    h, w = img.shape[0:2]
    crop_pad_size = config['datasets']['train'].get('gt_size', 256)

    # 填充
    if h < crop_pad_size or w < crop_pad_size:
        pad_h = max(0, crop_pad_size - h)
        pad_w = max(0, crop_pad_size - w)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    # 裁剪
    if img.shape[0] > crop_pad_size or img.shape[1] > crop_pad_size:
        h, w = img.shape[0:2]
        top = random.randint(0, h - crop_pad_size)
        left = random.randint(0, w - crop_pad_size)
        img = img[top:top + crop_pad_size, left:left + crop_pad_size, ...]

    # 生成退化核
    kernels = generate_kernels(config)

    # 应用退化
    degraded_tensor = apply_degradation(img, config, kernels, args.device)

    # 转换为numpy并保存
    degraded_img = tensor2img(degraded_tensor, rgb2bgr=True, out_type=np.uint8)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 保存结果
    cv2.imwrite(args.output, degraded_img)
    print(f"退化后的图像已保存到: {args.output}")

    # 可选：保存对比图
    comparison_path = args.output.replace('.png', '_comparison.png').replace('.jpg', '_comparison.jpg')
    if comparison_path != args.output:
        # 创建对比图
        h1, w1 = img.shape[:2]
        h2, w2 = degraded_img.shape[:2]
        max_h = max(h1, h2)
        comparison = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        comparison[:h1, :w1] = img
        comparison[:h2, w1:w1+w2] = degraded_img
        cv2.imwrite(comparison_path, comparison)
        print(f"对比图已保存到: {comparison_path}")


if __name__ == '__main__':
    main()

# python degrade_single_image.py \
#     --config options/finetune_realesrgan_x4plus.yml \
#     --input path/to/input/image.jpg \
#     --output path/to/output/degraded_image.jpg \
#     --device cuda \
#     --seed 42