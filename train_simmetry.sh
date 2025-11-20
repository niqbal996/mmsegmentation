#!/usr/bin/env bash
# --cfg-options load_from=/netscratch/naeem/swin_base_patch4_window12_384.pth \
export PYTHONPATH=$PYTHONPATH:/home/iqbal/mmsegmentation/
source ~/miniconda3/bin/activate
conda activate mmseg
pip install albumentations>=0.3.2 --no-binary qudida,albumentations
python3 tools/train.py \
configs/mask2former/mask2former_r50_simmetry-1024x1024.py \
--work-dir /netscratch/naeem/tfp_project/mask2former_amazone_p4ai_3_classes_AUGMENTED
