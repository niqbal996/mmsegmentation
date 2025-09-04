srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=10 -p RTXA6000 --mem=50000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/mmseg_forked:/home/iqbal/mmsegmentation,/home/iqbal/mmdetection:/home/iqbal/mmdetection,/home/iqbal/mmengine:/home/iqbal/mmengine \
  --container-image=/netscratch/naeem/mmseg_23.09_09_2025_ADA.sqsh  \
  --container-workdir=/home/iqbal \
  --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=mmseg_ADA \
  --time=00-08:00 \
  bash mmsegmentation/train_ada.sh