#!/usr/bin/env bash
python3 tools/train.py configs/hrnet/fcn_hr18_4xb2-40k_phenobench-1024x1024_phenobench.py --work-dir /netscratch/naeem/mmseg_output/wacv_results/HRNet_18_phenobench_vanilla
