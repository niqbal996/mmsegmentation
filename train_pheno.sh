#!/usr/bin/env bash
# python3 -m pip install pytest-runner --upgrade
# python3 -m pip install ftfy
# --cfg-options load_from=/netscratch/naeem/swin_base_patch4_window12_384.pth \
# nvidia-smi
#sed -i '283s/^/#/' /root/miniconda3/envs/mmseg/lib/python3.8/site-packages/mmengine/runner/loops.py
#sed -i '284s/^/#/' /root/miniconda3/envs/mmseg/lib/python3.8/site-packages/mmengine/runner/loops.py
sed -i "102s|.*|                img_bytes, flag=self.color_type, backend='pillow')|" /root/miniconda3/envs/mmseg/lib/python3.8/site-packages/mmcv/transforms/loading.py
export PYTHONPATH=$PYTHONPATH:/home/iqbal/mmsegmentation/
source ~/miniconda3/bin/activate
conda activate mmseg
python3 tools/train.py \
configs/pidnet/pidnet-s_1024x1024_al.py \
--work-dir /netscratch/naeem/tfp_project/pidnet_s_1024x1024_pheno_al
