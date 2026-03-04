#!/bin/bash
#SBATCH --job-name=mmseg_subset_training
#SBATCH --partition=RTXA6000
#SBATCH --mem=60G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mail-type=END
#SBATCH --mail-user=naeem.iqbal@dfki.de

# This script takes three arguments:
# 1. The config file path
# 2. The work directory (output directory)
# 3. REAL_SUBSET_RATIO value

CONFIG_FILE=$1
WORK_DIR=$2
REAL_SUBSET_RATIO=$3

if [ -z "$CONFIG_FILE" ] || [ -z "$WORK_DIR" ] || [ -z "$REAL_SUBSET_RATIO" ]; then
  echo "Usage: $0 <config_file> <work_dir> <real_subset_ratio>"
  exit 1
fi

mkdir -p "$WORK_DIR"

echo "✅ Starting training job for config: $CONFIG_FILE"
echo "✅ Output will be saved to: $WORK_DIR"
echo "✅ REAL_SUBSET_RATIO: $REAL_SUBSET_RATIO"

srun \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/mmsegmentation:/home/iqbal/mmsegmentation,/ds/images/cropandweed:/ds/images/cropandweed,/home/iqbal/mmengine:/home/iqbal/mmengine,/home/iqbal/mmdetection:/home/iqbal/mmdetection \
  --container-image=/netscratch/naeem/mmseg_23.09_09_2025_ADA.sqsh \
  --container-workdir=/home/iqbal/mmsegmentation \
  --time=00-08:00 \
  bash -c "source ~/miniconda3/bin/activate && conda activate mmseg && pip install albumentations>=0.3.2 && export REAL_SUBSET_RATIO=${REAL_SUBSET_RATIO} && python3 tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --eval-after-training"
