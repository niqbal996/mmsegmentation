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

| Model Name | Config Name | Weed IoU | Crop IoU | Tiny Recall | Model Checkpoint Link |
|---|---|---:|---:|---:|---|
| DeepLabV3+ R50 (Synthetic Baseline) | [deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26.py](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26.py) | - | - | - | TBD |
| DeepLabV3+ R50 (Pheno Test) | [deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test.py](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test.py) | - | - | - | TBD |
| DeepLabV3+ R50 (OHEM Loss) | [deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test_ohem_loss.py](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test_ohem_loss.py) | - | - | - | TBD |
| DeepLabV3+ R50 (Focal Loss) | [deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test_focal.py](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test_focal.py) | - | - | - | TBD |
| DeepLabV3+ R50 (Focal + Class Weight) | [deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test_focal_class_weighted.py](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test_focal_class_weighted.py) | - | - | - | TBD |
| DeepLabV3+ R50 (Focal + Dilated ASPP) | [deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test_focal_loss_dilated.py](configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_sugarbeetsynthetic26_pheno_test_focal_loss_dilated.py) | - | - | - | TBD |
| SegFormer MiT-B5 (Synthetic CE) | [segformer_mit-b5_4xb1-30k_syn_v6-1024x1024_ce_loss_baseline.py](configs/segformer/segformer_mit-b5_4xb1-30k_syn_v6-1024x1024_ce_loss_baseline.py) | - | - | - | TBD |
| SegFormer MiT-B5 (Synthetic OHEM) | [segformer_mit-b5_4xb1-30k_syn_v6-1024x1024_ohem_loss_baseline.py](configs/segformer/segformer_mit-b5_4xb1-30k_syn_v6-1024x1024_ohem_loss_baseline.py) | - | - | - | TBD |
| Mask2Former R50 (Mixed) | [mask2former_r50_active_mixing-512x512.py](configs/mask2former/mask2former_r50_active_mixing-512x512.py) | - | - | - | TBD |

