#!/usr/bin/env bash

source ~/miniconda3/bin/activate
conda activate mmseg
# python3 tools/train.py \
# configs/pidnet/pidnet-s_1024x1024_al.py \
# --work-dir /netscratch/naeem/tfp_project/pidnet_s_1024x1024_pheno_al

python3 tools/test.py \
configs/mask2former/mask2former_r50_phenobench-512x512.py \
/netscratch/naeem/mmseg_output/Mask2Former_random_sampling_phenobench_100/best_mIoU_iter_30000.pth \
--work-dir /netscratch/naeem/mmseg_output/ijcai_results/pheno2cropandweed/ \
--show-dir /netscratch/naeem/mmseg_output/ijcai_results/pheno2cropandweed/

python3 tools/test.py \
configs/mask2former/mask2former_r50_phenobench-512x512.py \
/netscratch/naeem/mmseg_output/mask2former_r50_ade20k_synthetic2phenobench_baseline/best_mIoU_iter_5000.pth \
--work-dir /netscratch/naeem/mmseg_output/ijcai_results/synthetic2cropandweed/ \
--show-dir /netscratch/naeem/mmseg_output/ijcai_results/synthetic2cropandweed/

python3 tools/train.py \
configs/mask2former/mask2former_r50_phenobench_cropweed-512x512.py \
--work-dir /netscratch/naeem/mmseg_output/ijcai_results/cropandweed2phenobench/