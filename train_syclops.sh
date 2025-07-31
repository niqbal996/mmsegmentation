#!/usr/bin/env bash
/root/miniconda3/envs/mmseg/bin/python tools/train.py \
configs/phenobench.py \
--cfg-options load_from=/netscratch/naeem/mmseg_checkpoints/mask2former_r50_8xb2-160k_ade20k-512x512_20221204_000055-2d1f55f1.pth \
--work-dir /netscratch/naeem/mmseg_output/mask2former_r50_ade20k_phenobench_baseline_20k
