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

| Model Name | Train/val | config | Weed IoU | Crop IoU | Tiny Recall | Model Checkpoint Link |
|---|---|---|---:|---:|---:|---|
| DeepLabV3+ R50  | SS / Ph | [config](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test.py) | 47.54 | 89.90 | 14.34 | TBD |
| DeepLabV3+ R50  | SS / SS | [config](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26.py) | 25.45 | 85.97 | 77.81 | TBD |
| DeepLabV3+ R50  | Ph / Ph | [config](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test_ohem_loss.py) | 69.47 | 94.12 | 64.75 | TBD |
| DeepLabV3+ R50  | SS + Ph 1% / Ph | [config](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_combined_ohem_loss_test_real.py) | 63.34 | 94.22 | 59.19 | TBD |
| DeepLabV3+ R50  | SS + Ph 5% / Ph | [config](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_combined_ohem_loss_test_real.py) | 69.68 | 95.07 | 65.05 | TBD |
| DeepLabV3+ R50  | SS + Ph 10% / Ph | [config](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_combined_ohem_loss_test_real.py) | 70.41 | 95.32 | 73.94 | TBD |
| DeepLabV3+ R50  | SS + Ph 1% / SS | [config](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_combined_ohem_loss_test_syn.py) | 61.32 | 94.06 | 60.20 | TBD |
| DeepLabV3+ R50  | SS + Ph 5% / SS | [config](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_combined_ohem_loss_test_syn.py) | 67.56 | 94.96 | 73.74 | TBD |
| DeepLabV3+ R50  | SS + Ph 10% / SS | [config](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_combined_ohem_loss_test_syn.pyy) | 68.96 | 95.18 | 78.79 | TBD |

# TODO 

- [ ] Add model checkpoints for the respective configs.
- [ ] Add slurm config for each experiment set.