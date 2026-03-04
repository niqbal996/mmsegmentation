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
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6_pheno_test_ohem_loss.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_synthetic_on_phenobench_ohem_loss_baseline/"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6_pheno_test_augmented.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_synthetic_on_phenobench_ohem_loss_weighted_augmented/"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench_syn_copypaste.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_phenobench_with_syn_copy_pasting_ohem_loss_weighted/"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6_ohem_loss_dilated_masks.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_syn_ohem_loss_dilated_masks_9x9_3"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench_syn_copypaste_poisson.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_phenobench_with_syn_copy_pasting_ohem_loss_weighted_poisson/"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_combined_ohem_loss_test_real.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_phenobench_combined_b4_50_50_real_1_percent_test_real/"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_combined_ohem_loss_test_syn.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_phenobench_combined_b4_50_50_real_1_percent_test_syn/"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_combined_ohem_loss_test_real_pretrained.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_phenobench_combined_b4_50_50_real_1_percent_test_real_pretrained"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_combined_ohem_loss_test_syn_pretrained.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_phenobench_combined_b4_50_50_real_1_percent_test_syn_pretrained/"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench_ohem_loss.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_phenobench_ohem_loss"
    # "configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench_subset_ohem_loss.py;/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_phenobench_subset_ohem_loss/"
    # "configs/segformer/segformer_mit-b5_4xb1-30k_phenobench-1024x1024_ce_loss_baseline.py;/netscratch/naeem/mmseg_output/eccv_results/SegFormer_experiments/SegFormer_mit-b5_phenobench_ce_loss_baseline/"
    # "configs/segformer/segformer_mit-b5_4xb1-30k_phenobench-1024x1024_ohem_loss_baseline.py;/netscratch/naeem/mmseg_output/eccv_results/SegFormer_experiments/SegFormer_mit-b5_phenobench_ohem_loss_baseline/"
    # "configs/segformer/segformer_mit-b5_4xb1-30k_syn_v6-1024x1024_ce_loss_baseline.py;/netscratch/naeem/mmseg_output/eccv_results/SegFormer_experiments/SegFormer_mit-b5_synthetic_ce_loss_baseline/"
    # "configs/segformer/segformer_mit-b5_4xb1-30k_syn_v6-1024x1024_ohem_loss_baseline.py;/netscratch/naeem/mmseg_output/eccv_results/SegFormer_experiments/SegFormer_mit-b5_synthetic_ohem_loss_baseline/"
    "configs/segformer/segformer_mit-b5_4xb1-30k_phenobench-1024x1024_ohem_loss_subsets.py;/netscratch/naeem/mmseg_output/eccv_results/SegFormer_experiments/SegFormer_mit-b5_phenobench_ohem_loss_subsets/"
    "configs/segformer/segformer_mit-b5_4xb1-30k_mixed-1024x1024_ohem_loss_test_syn.py;/netscratch/naeem/mmseg_output/eccv_results/SegFormer_experiments/SegFormer_mit-b5_combined_phenobench_test_syn_ohem_loss/"
    "configs/segformer/segformer_mit-b5_4xb1-30k_mixed-1024x1024_ohem_loss_test_real.py;/netscratch/naeem/mmseg_output/eccv_results/SegFormer_experiments/SegFormer_mit-b5_combined_phenobench_test_real_ohem_loss/"
)

# REAL_SUBSET_RATIO sweep values
declare -a subset_ratios=(0.01 0.05 0.1)

# Set to true to run REAL_SUBSET_RATIO sweep, false to run config-only jobs.
SWEEP_REAL_SUBSET_RATIO=true

# Loop through the trainings array and schedule jobs
for training in "${trainings[@]}"; do
    IFS=';' read -r config_file work_dir <<< "$training"

    if [ "$SWEEP_REAL_SUBSET_RATIO" = true ]; then
        for ratio in "${subset_ratios[@]}"; do
            ratio_tag=${ratio/./p}
            ratio_work_dir="${work_dir%/}/real_subset_${ratio_tag}"
            job_name="$(basename "${work_dir%/}")_r${ratio_tag}"

            mkdir -p "$ratio_work_dir"

            echo "Scheduling training for config: $config_file"
            echo "Output directory: $ratio_work_dir"
            echo "REAL_SUBSET_RATIO: $ratio"

            sbatch \
                --job-name="$job_name" \
                --output="${ratio_work_dir}/slurm-%j.out" \
                --error="${ratio_work_dir}/slurm-%j.err" \
                sbatch_train_subset_ratio.sh "$config_file" "$ratio_work_dir" "$ratio"

            echo "--------------------------------------------------"
        done
    else
        base_work_dir="${work_dir%/}"
        job_name="$(basename "$base_work_dir")"

        mkdir -p "$base_work_dir"

        echo "Scheduling training for config: $config_file"
        echo "Output directory: $base_work_dir"
        echo "REAL_SUBSET_RATIO: not set (config-only run)"

        sbatch \
            --job-name="$job_name" \
            --output="${base_work_dir}/slurm-%j.out" \
            --error="${base_work_dir}/slurm-%j.err" \
            sbatch_train.sh "$config_file" "$base_work_dir"

        echo "--------------------------------------------------"
    fi
done

if [ "$SWEEP_REAL_SUBSET_RATIO" = true ]; then
    echo "All training jobs have been scheduled for REAL_SUBSET_RATIO sweep: ${subset_ratios[*]}"
else
    echo "All training jobs have been scheduled in config-only mode."
fi
