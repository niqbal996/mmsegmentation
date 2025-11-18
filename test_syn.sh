#!/usr/bin/env bash
source ~/miniconda3/bin/activate
conda activate mmseg
cd mmengine && python3 -m pip install -e . && cd ..
cd mmsegmentation && python3 -m pip install -e . && cd ..
cd mmdetection && python3 -m pip install -e . && cd ..
sed -i "102s|.*|                img_bytes, flag=self.color_type, backend='pillow')|" /root/miniconda3/envs/mmseg/lib/python3.8/site-packages/mmcv/transforms/loading.py
export PYTHONPATH=$PYTHONPATH:/home/iqbal/mmsegmentation/:/home/iqbal/mmdetection:/home/iqbal/mmengine
python3 tools/test.py \
configs/mask2former/mask2former_r50_phenobench_cropweed-512x512.py \
/netscratch/naeem/mmseg_output/Mask2Former_random_sampling_sugarbeet_syn_v6_100/best_mIoU_iter_10000.pth \
--work-dir /netscratch/naeem/mmseg_output/ijcai_results/synthetic2cropandweed/ \
--show-dir /netscratch/naeem/mmseg_output/ijcai_results/synthetic2cropandweed/