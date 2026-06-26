#!/usr/bin/env bash
source ~/miniconda3/bin/activate && conda activate mmseg
python3 generate_metrics.py --schedule-file schedule_trainings.sh --eval-config configs/wacv/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench.py --metrics-subdir phenobench_pixel_precision_10_pix_threshold_seed_2_exg --summary-output phenobench_pixel_precision_10_pix_threshold_seed_2_exg.json