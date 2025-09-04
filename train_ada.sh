#!/usr/bin/env bash
source ~/miniconda3/bin/activate
conda activate mmseg
cd mmengine && python3 -m pip install -e . && cd ..
cd mmsegmentation && python3 -m pip install -e . && cd ..
cd mmdetection && python3 -m pip install -e . && cd ..
sed -i "102s|.*|                img_bytes, flag=self.color_type, backend='pillow')|" /root/miniconda3/envs/mmseg/lib/python3.8/site-packages/mmcv/transforms/loading.py
export PYTHONPATH=$PYTHONPATH:/home/iqbal/mmsegmentation/:/home/iqbal/mmdetection:/home/iqbal/mmengine
python3 mmsegmentation/tools/train.py \
mmsegmentation/configs/mask2former/mask2former_r50_ADA_active_mixing-512x512.py \
--work-dir /netscratch/naeem/mmseg_output/Mask2Former_ADA_Synv6_to_Phenobench_region_labelling