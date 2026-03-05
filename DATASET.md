# Data preparation
The experiments were done using `Phenobench` and `SugarbeetSynthetic2026`. The dataset should be in following directory structure. 
```bash
./phenobench/train/
├── images
├── leaf_instances
├── leaf_visibility
├── plant_instances
├── plant_visibility
├── semantics
./phenobench/val/
├── images
├── leaf_instances
├── leaf_visibility
├── plant_instances
├── plant_visibility
├── semantics

./sugarbeetsynthetic2026/images/
├── train
│   ├── 0000.png
│   ├── 0001.png
│   ├── ........
│   ├── 4499.png
├── val
│   ├── 4500.png
│   ├── 4501.png
│   ├── ........
│   ├── 4999.png
./sugarbeetsynthetic2026/main_camera_annotations/semantics
├── train
│   ├── 0000.png
│   ├── 0001.png
│   ├── ........
│   ├── 4499.png
├── val
│   ├── 4500.png
│   ├── 4501.png
│   ├── ........
│   ├── 4999.png
```

Depending upon the experiment and the relevant model config file, the datasets are imported as below e.g. for mixed training using `configs/deeplabv3plus/deeplabv3plus_r50-d8_8xb1-30k_1024x1024_combined_ohem_loss_test_syn_pretrained.py`:

```python
_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/combined_sugarbeetsynthetic2026_phenobench_test_syn.py',
    '../_base_/default_runtime.py',
]
```
where the mixed dataset `combined_sugarbeetsynthetic2026_phenobench_test_syn.py` is imported. 

The paths to the datasets can be set in the respective dataset config e.g. for `combined_sugarbeetsynthetic2026_phenobench_test_syn.py` to do `Real` and `Synthetic` mixed training, the respective dataset root has to be copy pasted from above.  

```python
import os
synthetic_dataset_type = 'SyclopsDataset'
real_dataset_type = 'PhenobenchDataset'

data_root_synthetic = '/path/to/sugarbeetsynthetic2026/root/'
data_root_real = '/path/to/phenobench/root/'
```
