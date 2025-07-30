srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=20 -p A100-40GB --mem=50000 \
  --container-mounts=/netscratch/vtyagi:/netscratch/vtyagi,/home/vtyagi/mmsegmentation:/home/vtyagi/mmsegmentation,/ds/images/cropandweed:/ds/images/cropandweed \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.04-py3.sqsh \
  --container-save=/netscratch/vtyagi/mmsegmentation_container.sqsh \
  --container-workdir=/home/vtyagi/mmsegmentation \
  --time=00-02:00 \
  --immediate=300 \
  --pty /bin/bash