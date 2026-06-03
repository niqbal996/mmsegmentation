import os
import glob
import numpy as np
from PIL import Image

val_inst_dir = '/path/to/phenobench/root/val/plant_instances'
val_sem_dir = '/path/to/phenobench/root/val/semantics'

inst_files = sorted(glob.glob(os.path.join(val_inst_dir, '*.png')))

total_crops = 0
total_weeds = 0
total_unknown = 0

for inst_file in inst_files:
    sem_file = os.path.join(val_sem_dir, os.path.basename(inst_file))
    
    inst_map = np.array(Image.open(inst_file))
    sem_map = np.array(Image.open(sem_file))
    
    # Apply the same mapping as PhenobenchDatasetAL
    sem_map[sem_map == 3] = 1
    sem_map[sem_map == 4] = 2
    
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]
    
    for iid in inst_ids:
        mask = inst_map == iid
        labels = sem_map[mask]
        labels = labels[(labels == 1) | (labels == 2)]
        
        if len(labels) == 0:
            total_unknown += 1
            continue
            
        uniq, counts = np.unique(labels, return_counts=True)
        cls = uniq[np.argmax(counts)]
        
        if cls == 1:
            total_crops += 1
        elif cls == 2:
            total_weeds += 1

print(f"Total images: {len(inst_files)}")
print(f"Total crops: {total_crops}")
print(f"Total weeds: {total_weeds}")
print(f"Total unknown: {total_unknown}")
