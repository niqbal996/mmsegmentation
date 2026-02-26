#!/bin/bash
#SBATCH --job-name=mmseg_training
#SBATCH --partition=RTX3090
#SBATCH --mem=40G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=naeem.iqbal@dfki.de

# This script takes two arguments:
# 1. The config file path
# 2. The work directory (output directory)

CONFIG_FILE=$1
WORK_DIR=$2

# Check if arguments are provided
if [ -z "$CONFIG_FILE" ] || [ -z "$WORK_DIR" ]; then
  echo "Usage: $0 <config_file> <work_dir>"
  exit 1
fi

# Create the output directory for logs if it doesn't exist
mkdir -p "$WORK_DIR"

# Set Slurm output and error log paths inside the script
#SBATCH --output=${WORK_DIR}/slurm-%j.out
#SBATCH --error=${WORK_DIR}/slurm-%j.err

# Set dynamic job name based on the work directory
JOB_NAME=$(basename "$WORK_DIR")
#SBATCH --job-name=${JOB_NAME}

echo "✅ Starting training job for config: $CONFIG_FILE"
echo "✅ Output will be saved to: $WORK_DIR"

srun \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/mmsegmentation:/home/iqbal/mmsegmentation,/ds/images/cropandweed:/ds/images/cropandweed,/home/iqbal/mmengine:/home/iqbal/mmengine,/home/iqbal/mmdetection:/home/iqbal/mmdetection \
  --container-image=/netscratch/naeem/mmseg_23.09_09_2025_ADA.sqsh  \
  --container-workdir=/home/iqbal/mmsegmentation \
  --time=00-14:00 \
  bash -c "source ~/miniconda3/bin/activate && conda activate mmseg && pip install albumentations>=0.3.2 && python3 tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --eval-after-training"
