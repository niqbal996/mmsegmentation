import os
import os.path as osp
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import fileio
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from tools.minimal_inference import ImageProcessor

@DATASETS.register_module()
class PhenobenchDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crop', 'weed'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    )

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def load_data_list(self):
        """Load annotation from directory.
        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True):
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                seg_map = img
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = None
            data_info['reduce_zero_label'] = False
            data_info['seg_fields'] = []
            data_list.append(data_info)

        return data_list

    def load_annotations(self, img_path, seg_map_path):
        """Load annotation from png file.
        Args:
            img_path (str): Path to image file.
            seg_map_path (str): Path to segmentation png file.
        Returns:
            dict: The dict contains loaded image and semantic segmentation annotations.
        """
        img_info = dict(filename=img_path)
        seg_map = np.array(Image.open(seg_map_path))
        
        
        # Convert class 3 to 1 (crop) and class 4 to 2 (weed)
        seg_map[seg_map == 3] = 1
        seg_map[seg_map == 4] = 2
        
        img_info['gt_seg_map'] = seg_map
        return img_info

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        return self.get_data_info(idx)
    
@DATASETS.register_module()
class PhenobenchDatasetRegionBased(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crop', 'weed'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    )

    def __init__(self, region_percentage=0.1, labelling_round=1, 
                 active_mask_dir='semantics_active_mask',
                 active_indicator_dir='semantics_active_indicator',
                 **kwargs):
        self.region_percentage = region_percentage 
        self.labelling_round = labelling_round
        self.active_mask_dir = active_mask_dir
        self.active_indicator_dir = active_indicator_dir
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def load_data_list(self):
        """Load annotation from directory.
        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        subset_dir = osp.split(ann_dir)[0]
        active_mask_dir = osp.join(subset_dir, self.active_mask_dir)
        active_indicator_dir = osp.join(subset_dir, self.active_indicator_dir)

        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True):
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                seg_map = img
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = None
            data_info['reduce_zero_label'] = False
            data_info['seg_fields'] = []
            data_info['active_mask_path'] = osp.join(active_mask_dir, img)
            data_info['active_indicator_path'] = osp.join(active_indicator_dir, img)
            data_list.append(data_info)

        return data_list

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        return self.get_data_info(idx)
    
@DATASETS.register_module()
class PhenobenchDatasetAL(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crop', 'weed'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    )

    def __init__(self, subset_ratio=1.0, sample_list=None, **kwargs):
        self.subset_ratio = subset_ratio
        self.sample_list = sample_list
        
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def load_data_list(self):
        """Load annotation from directory.
        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        
        if self.subset_ratio < 1.0:
            # Load sample list if subset_ratio is less than 1.0
            sample_list_path = self.sample_list
            if sample_list_path is None:
                raise ValueError("sample_list must be provided when subset_ratio < 1.0")
            with open(sample_list_path, 'r') as f:
                sample_list = [line.strip().rsplit(',', 1)[0] for line in f.readlines()]
            data_iterator = sample_list
        else:
            data_iterator = fileio.list_dir_or_file(
                                                    dir_path=img_dir,
                                                    list_dir=False,
                                                    suffix=self.img_suffix,
                                                    recursive=True)
            
        for img in data_iterator:
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                seg_map = img
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = None
            data_info['reduce_zero_label'] = False
            data_info['seg_fields'] = []
            data_list.append(data_info)

        return data_list

    def load_annotations(self, img_path, seg_map_path):
        """Load annotation from png file.
        Args:
            img_path (str): Path to image file.
            seg_map_path (str): Path to segmentation png file.
        Returns:
            dict: The dict contains loaded image and semantic segmentation annotations.
        """
        img_info = dict(filename=img_path)
        seg_map = np.array(Image.open(seg_map_path))
        
        # Convert class 3 to 1 (crop) and class 4 to 2 (weed)
        seg_map[seg_map == 3] = 1
        seg_map[seg_map == 4] = 2
        
        img_info['gt_seg_map'] = seg_map
        return img_info

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        return self.get_data_info(idx)
    
@DATASETS.register_module()
class PhenoBench_processed(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crop', 'weed'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    )

    def __init__(self, **kwargs):
        self.processor = ImageProcessor()
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def load_data_list(self):
        """Load annotation from directory.
        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True):
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                seg_map = img
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = None
            data_info['reduce_zero_label'] = False
            data_info['seg_fields'] = []
            data_list.append(data_info)

        return data_list

    def load_annotations(self, img_path, seg_map_path):
        """Load annotation from png file.
        Args:
            img_path (str): Path to image file.
            seg_map_path (str): Path to segmentation png file.
        Returns:
            dict: The dict contains loaded image and semantic segmentation annotations.
        """
        img_info = dict(filename=img_path)
        seg_map = np.array(Image.open(seg_map_path))
        
        
        # Convert class 3 to 1 (crop) and class 4 to 2 (weed)
        seg_map[seg_map == 3] = 1
        seg_map[seg_map == 4] = 2
        
        img_info['gt_seg_map'] = seg_map
        return img_info

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        return self.get_data_info(idx)
    
    def __getitem__(self, idx):
        # ...existing code...
        data_info = self.get_data_info(idx)
        img_path = data_info['img_path']
        mask_path = data_info.get('seg_map_path', None)
        img_tensor, img_meta = self.processor.preprocess(img_path)

        mask = None
        gt_sem_seg = None
        if mask_path is not None:
            mask = np.array(Image.open(mask_path)).astype(np.uint8)
            mask[mask == 3] = 1
            mask[mask == 4] = 2
            _, _, h, w = img_tensor.shape
            pad_h = h - mask.shape[0]
            pad_w = w - mask.shape[1]
            if pad_h > 0 or pad_w > 0:
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)  # [1, H, W]
            gt_sem_seg = PixelData(data=mask_tensor, metainfo=img_meta)

        # --- Active Learning additions ---
        # These are optional and only added if files exist in data_info
        origin_mask = None
        origin_label = None
        active_indicator = None
        active_selected = None
        seg_map_parts = Path(data_info['seg_map_path']).parts
        seg_mask = seg_map_parts[-1]
        seg_map_dir = os.path.join(*seg_map_parts[:-2])
        path_to_mask = data_info['seg_map_path']
        indicator_dir = osp.join(seg_map_dir, 'semantics_indicator_mask')
        os.makedirs(indicator_dir, exist_ok=True)
        path_to_indicator = osp.join(indicator_dir, seg_mask)
        size = None
        if mask is not None:
            origin_mask = torch.from_numpy(mask).long()
            origin_label = torch.from_numpy(mask).long()
            size = torch.tensor([mask.shape[0], mask.shape[1]])
            # Try to load indicator if available
            if path_to_indicator is not None and os.path.exists(path_to_indicator):
                indicator = torch.load(path_to_indicator)
                active_indicator = indicator['active']
                active_selected = indicator['selected']
            else:
                active_indicator = torch.zeros_like(origin_mask, dtype=torch.bool)
                active_selected = torch.zeros_like(origin_mask, dtype=torch.bool)

        ret = {
            'inputs': img_tensor.squeeze(0),  # [C, H, W]
            'gt_sem_seg': gt_sem_seg,
            'metainfo': img_meta,
            'img_path': img_path,
            'mask_path': mask_path
        }
        # Add active learning tensors if available
        if origin_mask is not None:
            ret['origin_mask'] = origin_mask
        if origin_label is not None:
            ret['origin_label'] = origin_label
        if active_indicator is not None:
            ret['active'] = active_indicator
        if active_selected is not None:
            ret['selected'] = active_selected
        if path_to_mask is not None:
            ret['path_to_mask'] = path_to_mask
        if path_to_indicator is not None:
            ret['path_to_indicator'] = path_to_indicator
        if size is not None:
            ret['size'] = size

        return ret