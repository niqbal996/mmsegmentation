#!/bin/bash

# An array of training configurations and their output directories
# Each element is a string with "config_file;work_dir"
declare -a trainings=(
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_phenobench"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_synthetic_baseline"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6_pheno_test.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_synthetic_on_phenobench_baseline/"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6_pheno_test_focal.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_synthetic_on_phenobench_focal_loss_baseline/"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6_pheno_test_focal_class_weighted.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_synthetic_on_phenobench_focal_loss_class_weighted/"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6_pheno_test_focal_loss_dilated.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_synthetic_on_phenobench_focal_loss_dilated/"
    "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6_pheno_test_ohem_loss.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_synthetic_on_phenobench_ohem_loss_baseline/"
    "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6_pheno_test_augmented.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_synthetic_on_phenobench_focal_loss_weighted_augmented/"
    "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench_syn_copypaste.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_phenobench_with_syn_copy_pasting_focal_loss_weighted/"
)

# Loop through the trainings array and schedule each job
for training in "${trainings[@]}"; do
    IFS=';' read -r config_file work_dir <<< "$training"
    
    echo "Scheduling training for config: $config_file"
    echo "Output directory: $work_dir"
    
    sbatch sbatch_train.sh "$config_file" "$work_dir"
    
    echo "--------------------------------------------------"
done

echo "All training jobs have been scheduled."
