CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=7 --master_port=4321 realesrgan/train.py -opt options/train_realesrnet_x1plus_yh.yml --launcher pytorch