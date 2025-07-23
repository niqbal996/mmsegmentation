import os.path as osp
import numpy as np
from PIL import Image

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import fileio

@DATASETS.register_module()
class CropAndWeedDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'sugarbeet', 'weed'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    )

    def __init__(self, variant='SugarBeet2', **kwargs):
        self.variant = variant
        
        super().__init__(
            img_suffix='.jpg',
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
        ann_dir = osp.join(ann_dir, self.variant)
        
        # Iterate through annotation files first to get only the subset
        for seg_map in fileio.list_dir_or_file(
                dir_path=ann_dir,
                list_dir=False,
                suffix=self.seg_map_suffix,
                recursive=True):
            # Look for corresponding image file
            img_path = osp.join(img_dir, seg_map)
            img_path = img_path.replace(self.seg_map_suffix, self.img_suffix)
            # Check if the corresponding image exists
            if osp.exists(img_path):
                data_info = dict(img_path=img_path)
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = None
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)

        return data_list

    # def load_annotations(self, img_path, seg_map_path):
    #     """Load annotation from png file.
    #     Args:
    #         img_path (str): Path to image file.
    #         seg_map_path (str): Path to segmentation png file.
    #     Returns:
    #         dict: The dict contains loaded image and semantic segmentation annotations.
    #     """
    #     img_info = dict(filename=img_path)
    #     seg_map = np.array(Image.open(seg_map_path))
        
    #     # Convert class 3 to 1 (crop) and class 4 to 2 (weed)
    #     # seg_map[seg_map == 3] = 1
    #     # seg_map[seg_map == 4] = 2
        
    #     img_info['gt_seg_map'] = seg_map
    #     return img_info

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        return self.get_data_info(idx)