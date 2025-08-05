# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models

import sys

# 将 --local-rank 转换为 --local_rank
for i, arg in enumerate(sys.argv):
    if arg.startswith('--local-rank='):
        value = arg.split('=')[1]
        sys.argv[i] = f'--local_rank={value}'
    elif arg == '--local-rank' and i+1 < len(sys.argv):
        sys.argv[i] = '--local_rank'

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
