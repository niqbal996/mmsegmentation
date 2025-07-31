#!/usr/bin/env bash

# Get subset percentage from argument
subset_percentage=$1

# Check if subset_percentage was passed
if [[ -z "$subset_percentage" ]]; then
  echo "❌ ERROR: Subset percentage not provided!"
  exit 1
fi

# Convert percentage to ratio (e.g., 1 → 0.01)
subset_ratio=$(awk "BEGIN {printf \"%.4f\", $subset_percentage / 100}")

export SUBSET_RATIO=$subset_ratio
# Set unlimited stack size
ulimit -s unlimited

echo "✅ Subset Index (percentage): $subset_percentage%"
echo "✅ Converted Subset Ratio: $subset_ratio"

# Define dynamic output directory with subset percentage
output_dir="/netscratch/naeem/mmseg_output/Mask2Former_random_sampling_phenobench_${subset_percentage}"
echo "✅ Output directory: $output_dir"
sed -i "102s|.*|                img_bytes, flag=self.color_type, backend='pillow')|" /root/miniconda3/envs/mmseg/lib/python3.8/site-packages/mmcv/transforms/loading.py
export PYTHONPATH=$PYTHONPATH:/home/iqbal/mmsegmentation/
source ~/miniconda3/bin/activate
conda activate mmseg
python3 tools/train.py \
configs/mask2former/mask2former_r50_phenobench-512x512.py \
--work-dir $output_dir