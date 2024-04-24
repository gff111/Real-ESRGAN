#!/bin/sh
# DATA_DIR=/data/yh/qrestore_dataset
# OUTPUT_DIR=/data/yh/train_local_log/qrestore_mobile_ngf_8_dgf_8_no_norm
mkdir -p /root/.pip/
cat > /root/.pip/pip.conf << EOF
[global]
index-url = http://jfrog.cloud.qiyi.domain/api/pypi/pypi/simple
trusted-host = jfrog.cloud.qiyi.domain
extra-index-url = http://jfrog.cloud.qiyi.domain/api/pypi/iqiyi-pypi-mesos/simple
EOF


# pip install ${PWD}/timm-0.9.11-py3-none-any.whl

PWD=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
# pip install ${PWD}/timm-0.9.11-py3-none-any.whl
# pip install --no-index --find-link=${DATA_DIR}/repodata -r ${PWD}/requirement.txt
sh ${PWD}/pip_envs.sh
ls -l $PWD
echo ${DATA_DIR}
echo ${OUTPUT_DIR}

# ls ${DATA_DIR}/df2k_ost_dataset/hr -lhtr|wc -l
# nvidia-smi
# python change_path_in_json_for_jarvis.py --input_file ${PWD}/options/swinir/train_swinir_sr_realworld_x2_psnr_1124_unshuffle_v2_sim_v2_bsrgan_plus.json --output_file ${PWD}/options/swinir/train_swinir_sr_realworld_x2_psnr_1124_unshuffle_v2_sim_v2_bsrgan_plus_chang.json --dataroot_H ${DATA_DIR}/df2k_ost_dataset/hr --log_root_path ${OUTPUT_DIR}

# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt ${PWD}/options/swinir/train_swinir_sr_realworld_x2_psnr_1124_unshuffle_v2_sim_v2_bsrgan_plus_chang.json --dist True

# ls ${DATA_DIR}/train_hdr -lhtr|wc -l
# nvidia-smi
# python change_path_in_json_for_jarvis_s2h.py --input_file ${PWD}/options/deart/swin_v2_degrad_bsrgan_x1_wo_resize.json --output_file ${PWD}/options/deart/swin_v2_degrad_bsrgan_x1_wo_resize_change.json --train_dataroot_H ${DATA_DIR}/df2k_ost_dataset/hr --test_dataroot_H ${DATA_DIR}/youku_deart_testset/HR --test_dataroot_L ${DATA_DIR}/youku_deart_testset/LR_bsrgab_x1_wo_resize --log_root_path ${OUTPUT_DIR}

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=0123 main_train_psnr.py --opt ${PWD}/options/deart/swin_v2_degrad_bsrgan_x1_wo_resize_change.json  --dist True
