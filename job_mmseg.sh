srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=10 -p RTXA6000 --mem=60000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/mmsegmentation:/home/iqbal/mmsegmentation,/ds/images/gta5/processed/:/ds/images/gta5 \
  --container-image=/netscratch/naeem/mmseg_pytorch_25.03.sqsh \
  --container-workdir=/home/iqbal/mmsegmentation \
  --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=HRNet_vanilla \
  --time=01-00:00 \
  bash train_model.sh
