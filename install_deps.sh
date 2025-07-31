# #!/usr/bin/env bash
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
conda create --name mmseg python=3.8 -y
conda activate mmseg
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.0"
pip install "mmdet>=3.0.0rc4"
pip install ftfy regex future tensorboard
pip install -v -e .