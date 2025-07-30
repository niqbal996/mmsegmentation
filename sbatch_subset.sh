#!/bin/bash
#SBATCH --job-name phenobench_mmseg_random_sampling
#SBATCH --partition=RTXA6000
#SBATCH --mem=40G                  # 🟢 Per task memory
#SBATCH --cpus-per-task=5          # 🟢 Per task CPU
#SBATCH --gres=gpu:1               # 🟢 Per task GPU
#SBATCH --ntasks=1                 # 🟢 1 task per array job
#SBATCH --array=0-7                # 🟢 Run 8 jobs in parallel (indices 0-7)
#SBATCH --chdir=/netscratch/naeem/mmseg_output/
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=naeem.iqbal@dfki.de

# ⚡ Define subset percentages
subset_indices=(1 2 5 10 20 40 80 100)

# Get the percentage for this SLURM array job
subset_percentage=${subset_indices[$SLURM_ARRAY_TASK_ID]}

# Set dynamic job name
SLURM_JOB_NAME="Mask2Former_subset_phenobench_${subset_percentage}"
echo "✅ Running job for subset percentage: $subset_percentage%"

# Pass subset_percentage to train_subset.sh
srun \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/mmsegmentation:/home/iqbal/mmsegmentation,/ds/images/cropandweed:/ds/images/cropandweed \
  --container-image=/netscratch/naeem/mmseg_23.09_07_2025.sqsh  \
  --container-workdir=/home/iqbal/mmsegmentation \
  --time=01-00:00 \
  bash train_subset.sh $subset_percentage