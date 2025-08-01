import os
import cv2
import numpy as np
import math
import torch
import random
from glob import glob
from tqdm import tqdm
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_gaussian_noise(img, sigma=10, gray=False):
    noise = np.random.randn(*img.shape[:2]) * sigma / 255.
    if gray or img.ndim == 2 or img.shape[2] == 1:
        noise = np.stack([noise] * 3, axis=2)
    else:
        noise = np.random.randn(*img.shape) * sigma / 255.
    return (img + noise).clip(0, 1)

def add_poisson_noise(img, scale=1.0, gray=False):
    img_scaled = img * 255.
    img_scaled = np.clip(img_scaled, 0, 255)
    img_scaled = np.nan_to_num(img_scaled, nan=0.0, posinf=255.0, neginf=0.0)

    if gray:
        lam = img_scaled.mean(axis=2, keepdims=True) * scale
        lam = np.clip(lam, 0, None)
        noise = np.random.poisson(lam) / 255. / scale
        noise = np.repeat(noise, 3, axis=2)
    else:
        lam = img_scaled * scale
        lam = np.clip(lam, 0, None)
        noise = np.random.poisson(lam) / 255. / scale
    return np.clip(noise, 0, 1)

def random_resize(img, resize_prob, resize_range):
    up, down, keep = resize_prob
    r = np.random.rand()
    if r < up:
        scale = np.random.uniform(1, resize_range[1])
    elif r < up + down:
        scale = np.random.uniform(resize_range[0], 1)
    else:
        scale = 1
    interpolation = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    new_size = (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale)))
    return cv2.resize(img, new_size, interpolation=interpolation)

def random_blur(img):
    kernel_size = random.choice(kernel_range)
    desc = ""

    if np.random.rand() < sinc_prob:
        omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=blur_kernel_size)
        desc = f"Sinc kernel: omega_c={omega_c:.2f}, size={kernel_size}"
    else:
        try:
            kernel_type = random.choices(kernel_list, kernel_prob)[0]
            sigma_x = np.random.uniform(*blur_sigma)
            sigma_y = np.random.uniform(*blur_sigma)
            rotation = np.random.uniform(-math.pi, math.pi)
            betag = np.random.uniform(*betag_range)
            betap = np.random.uniform(*betap_range)
            epsilon = 1e-6
            kernel = random_mixed_kernels(
                kernel_list=[kernel_type],
                kernel_prob=[1.0],
                kernel_size=kernel_size,
                sigma_x_range=[sigma_x, sigma_x + epsilon],
                sigma_y_range=[sigma_y, sigma_y + epsilon],
                rotation_range=[rotation, rotation + epsilon],
                betag_range=[betag, betag + epsilon],
                betap_range=[betap, betap + epsilon],
                noise_range=None
            )
            pad = (blur_kernel_size - kernel_size) // 2
            kernel = np.pad(kernel, ((pad, pad), (pad, pad)))
            desc = f"{kernel_type} kernel: size={kernel_size}, sigma_x={sigma_x:.2f}, sigma_y={sigma_y:.2f}, rotation={rotation:.2f}, betag={betag:.2f}, betap={betap:.2f}"
        except Exception as e:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=blur_kernel_size)
            desc = f"[Fallback] Sinc kernel: omega_c={omega_c:.2f}, size={kernel_size}, error={str(e)}"

    return cv2.filter2D(img, -1, kernel), desc

def add_random_noise(img, gauss_prob, noise_range, poisson_range, gray_prob):
    if np.random.rand() < gauss_prob:
        sigma = np.random.uniform(*noise_range)
        gray = np.random.rand() < gray_prob
        return add_gaussian_noise(img, sigma, gray), f"Gaussian Noise: sigma={sigma:.2f}, gray={gray}"
    else:
        scale = np.random.uniform(*poisson_range)
        gray = np.random.rand() < gray_prob
        return add_poisson_noise(img, scale, gray), f"Poisson Noise: scale={scale:.2f}, gray={gray}"

def jpeg_compress(img, jpeg_range):
    quality = random.randint(*jpeg_range)
    _, encimg = cv2.imencode('.jpg', (img * 255).clip(0, 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255., quality

def center_crop(img, size):
    h, w = img.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return img[top:top+size, left:left+size]

def degrade(img):
    log = []
    img = img.astype(np.float32) / 255.

    img, desc = random_blur(img)
    log.append(f"Blur1: {desc}")

    img = random_resize(img, resize_prob, resize_range)
    log.append(f"Resize1: prob={resize_prob}, range={resize_range}")

    img, noise_desc = add_random_noise(img, gaussian_noise_prob, noise_range, poisson_scale_range, gray_noise_prob)
    log.append(noise_desc)

    img, q1 = jpeg_compress(img, jpeg_range)
    log.append(f"JPEG Compression1: quality={q1}")

    if np.random.rand() < second_blur_prob:
        img, desc = random_blur(img)
        log.append(f"Blur2: {desc}")

    img = random_resize(img, resize_prob2, resize_range2)
    log.append(f"Resize2: prob={resize_prob2}, range={resize_range2}")

    img, noise_desc = add_random_noise(img, gaussian_noise_prob2, noise_range2, poisson_scale_range2, gray_noise_prob2)
    log.append(noise_desc)

    img, q2 = jpeg_compress(img, jpeg_range2)
    log.append(f"JPEG Compression2: quality={q2}")

    img = center_crop(img, gt_size)
    return img, "<br>".join(log)

def process_and_save_images(input_dirs, output_root, num_images=100):
    os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)
    html_lines = [
        '<html><head><meta charset="UTF-8"><title>退化图像对比</title></head><body>',
        '<h1>退化图像对比（原图 vs 退化图）</h1>',
        '<style>img{width:256px;} td{vertical-align:top;}</style>',
        '<table border="1" cellspacing="0" cellpadding="5">',
        '<tr><th>原图</th><th>退化图</th><th>退化过程</th></tr>'
    ]

    count = 0
    for input_dir in input_dirs:
        all_images = sorted(glob(os.path.join(input_dir, '*')))
        sampled = random.sample(all_images, min(num_images, len(all_images)))
        for img_path in tqdm(sampled, desc=f'Processing {input_dir}'):
            img_name = f'{count:04d}_{os.path.basename(img_path)}'
            img = cv2.imread(img_path)
            if img is None or img.shape[0] < gt_size or img.shape[1] < gt_size:
                continue
            gt = center_crop(img, gt_size)
            lq, log_str = degrade(gt.copy())
            gt_path = os.path.join(output_root, 'images', f'{img_name}_gt.png')
            lq_path = os.path.join(output_root, 'images', f'{img_name}_lq.png')
            cv2.imwrite(gt_path, gt)
            cv2.imwrite(lq_path, (lq * 255).clip(0, 255).astype(np.uint8))
            html_lines.append(
                f'<tr><td><img src="images/{img_name}_gt.png"></td>'
                f'<td><img src="images/{img_name}_lq.png"></td>'
                f'<td><pre>{log_str}</pre></td></tr>'
            )
            count += 1

    html_lines.append('</table></body></html>')
    html_path = os.path.join(output_root, 'index_poster.html')
    with open(html_path, 'w') as f:
        f.write('\n'.join(html_lines))
    print(f'HTML 已保存至: {html_path}')

# 参数定义
resize_prob = [0.2, 0.7, 0.1]
resize_range = [0.8, 1.5]
gaussian_noise_prob = 0.5
noise_range = [1, 30]
poisson_scale_range = [0.05, 3]
gray_noise_prob = 0.0
jpeg_range = [60, 95]

second_blur_prob = 0.8
resize_prob2 = [0.3, 0.4, 0.3]
resize_range2 = [0.8, 1.2]
gaussian_noise_prob2 = 0.5
noise_range2 = [1, 25]
poisson_scale_range2 = [0.05, 2.5]
gray_noise_prob2 = 0.0
jpeg_range2 = [60, 95]

gt_size = 512
kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
kernel_prob = [0.80, 0.2, 0.0, 0.0, 0.0, 0.0]
blur_kernel_size = 7
blur_sigma = [0.2, 1]
betag_range = [0.5, 4]
betap_range = [1, 2]
sinc_prob = 0.1
kernel_range = [3, 5, 7]

if __name__ == '__main__':
    set_random_seed(20250723)  # 固定随机种子
    input_dirs = [
        # '/root/paddlejob/workspace/qiucan/datasets/sa_text_high',
        # '/root/paddlejob/workspace/qiucan/datasets/RealCE/train/52mm'
        "/root/paddlejob/workspace/qiucan/datasets/poster_text_images_high"
        # "/root/paddlejob/workspace/env_run/zhuyinghao/datasets/dataset_lite/aigc_ocr_part1_img"
        # "/root/paddlejob/workspace/qiucan/datasets/Flickr2K_HR"
    ]
    output_root = '/root/paddlejob/workspace/env_run/zhuyinghao/Real-ESRGAN/results/degrade_res_poster'
    process_and_save_images(input_dirs, output_root, num_images=100)