
export CUDA_VISIBLE_DEVICES=4,5,6,7
# 1. train realesrnet_x1plus_yh
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4plus_yh.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4plus_yh60.yml --launcher pytorch

# 2. train realesrgan_x1plus_yh
# python -m torch.distributed.launch --nproc_per_node=7 --master_port=4321 realesrgan/train.py -opt options/train_realesrgan_x1plus_yh.yml --launcher pytorch