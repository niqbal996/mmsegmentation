srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=10 -p RTX3090 --mem=50000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/mmsegmentation:/home/iqbal/mmsegmentation,/home/iqbal/mmdetection:/home/iqbal/mmdetection,/home/iqbal/mmengine:/home/iqbal/mmengine,/ds/images/cropandweed:/ds/images/cropandweed \
  --container-image=/netscratch/naeem/mmseg_23.09_09_2025_ADA.sqsh  \
  --container-workdir=/home/iqbal/mmsegmentation \
  --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=phenobench_2_class \
  --time=00-08:00 \
  bash train_syclops.sh