srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=10 -p RTX3090 --mem=50000 \
  --container-image=/netscratch/naeem/mmseg_23.09_07_2025.sqsh  \
  --container-mounts=/netscratch/software:/netscratch/software:ro,/netscratch/naeem:/netscratch/naeem,/home/iqbal/mmsegmentation:/home/iqbal/mmsegmentation,/ds/images/cropandweed:/ds/images/cropandweed/,"`pwd`":"`pwd`"  \
  --container-workdir="`pwd`" \
  --time=00-04:00 \
  start_code_server.sh