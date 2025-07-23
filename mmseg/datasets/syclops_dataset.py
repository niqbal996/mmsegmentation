from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import fileio
import numpy as np
import os.path as osp
import random

@DATASETS.register_module()
class CustomSegDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crop', 'weed', 'other'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]]
    )

    def __init__(self, subset_fraction=1.0, random_seed=None, **kwargs):
        """Initialize the dataset.
        
        Args:
            subset_fraction (float): Fraction of dataset to use (between 0 and 1)
            random_seed (int, optional): Random seed for reproducibility
            **kwargs: Additional arguments passed to BaseSegDataset
        """
        self.subset_fraction = max(0.0, min(1.0, subset_fraction))  # Clip between 0 and 1
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        
    def load_data_list(self):
        """Load annotation from directory and optionally subset the data.
        Returns:
            list[dict]: Selected data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        
        # First collect all data
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True):
            data_info = dict(
                img_path=osp.join(img_dir, img),
                seg_fields=[],
                sample_idx=len(data_list)
            )
            if ann_dir is not None:
                seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = None
            data_info['reduce_zero_label'] = False
            data_info['seg_fields'] = []
            data_list.append(data_info)
        
        # Then subset the data if needed
        if self.subset_fraction < 1.0:
            num_samples = len(data_list)
            subset_size = int(num_samples * self.subset_fraction)
            data_list = random.sample(data_list, subset_size)
            
            # Update sample indices
            for i, data_info in enumerate(data_list):
                data_info['sample_idx'] = i
                
            print(f'Using {subset_size}/{num_samples} samples ({self.subset_fraction:.1%})')
        
        return data_list

    def load_annotations(self, img_path, seg_map_path):
        """Load annotation from npz file.
        Args:
            img_path (str): Path to image file.
            seg_map_path (str): Path to segmentation npz file.
        Returns:
            dict: The dict contains loaded image and semantic segmentation annotations.
        """
        img_info = dict(filename=img_path, seg_fields=[])
        seg_map = np.load(seg_map_path)['array']
        
        # Ensure the segmentation map is in the correct format (H, W)
        if seg_map.ndim != 2:
            raise ValueError(f"Segmentation map should be 2D, got shape {seg_map.shape}")
        
        img_info['gt_seg_map'] = seg_map
        img_info['seg_fields'].append('gt_seg_map')
        return img_info

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        return self.load_annotations(self.data_list[idx]['img_path'],
                                   self.data_list[idx]['seg_map_path'])