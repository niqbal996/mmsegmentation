# Installation 
For MMSeg installation, please follow the official mmseg installation [instructions](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).  

For dataset preparation and config paths, please refer to [datasets](DATASET.md).
# Training
For any experiment, the relevant config needs to be provided to the `tools/train.py` script. 
```bash
python3 tools/train.py --config-file configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6.py 
```
For mixed training, the subset ratio has to be provided as `REAL_SUBSET_RATIO` environment variable. 
```bash
REAL_SUBSET_RATIO=0.1 python3 tools/train.py --config-file configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_phenobench_subset_ohem_loss.py
```

# Testing
```bash
python3 tools/train.py --config-file configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_syn_v6.py 
--eval-after-training
```

# Model Zoo 

