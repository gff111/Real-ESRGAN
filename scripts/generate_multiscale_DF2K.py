import argparse
import glob
import os
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
import sys


def worker(args):
    """Worker for each process.

    Args:
        args: tuple containing (path, scale_list, shortest_edge, output_folder)

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    path, scale_list, shortest_edge, output_folder = args
    try:
        basename = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path)
        width, height = img.size

        # Generate multi-scale versions
        for idx, scale in enumerate(scale_list):
            rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
            rlt.save(os.path.join(output_folder, f'{basename}T{idx}.png'))

        # save the smallest image which the shortest edge is 400
        if width < height:
            ratio = height / width
            width = shortest_edge
            height = int(width * ratio)
        else:
            ratio = width / height
            height = shortest_edge
            width = int(height * ratio)
        rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        rlt.save(os.path.join(output_folder, f'{basename}T{len(scale_list)}.png'))

        return f'Processed {basename}'
    except Exception as e:
        return f'Error processing {path}: {str(e)}'


def main(args):
    """Generate multi-scale versions for GT images with LANCZOS resampling.
    It is now used for DF2K dataset (DIV2K + Flickr 2K)
    """
    # For DF2K, we consider the following three scales,
    # and the smallest image whose shortest edge is 400
    scale_list = [0.5]
    shortest_edge = 512

    path_list = sorted(glob.glob(os.path.join(args.input, '*')))

    # Prepare arguments for multiprocessing
    worker_args = [(path, scale_list, shortest_edge, args.output) for path in path_list]

    # Use multiprocessing pool
    pbar = tqdm(total=len(worker_args), unit='image', desc='Generate')
    pool = Pool(args.n_thread)

    for result in pool.imap_unordered(worker, worker_args):
        pbar.update(1)

    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


if __name__ == '__main__':
    """Generate multi-scale versions for GT images with LANCZOS resampling.
    It is now used for DF2K dataset (DIV2K + Flickr 2K)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/DF2K/DF2K_HR', help='Input folder')
    parser.add_argument('--output', type=str, default='datasets/DF2K/DF2K_multiscale', help='Output folder')
    parser.add_argument('--n_thread', type=int, default=8, help='Thread number.')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
