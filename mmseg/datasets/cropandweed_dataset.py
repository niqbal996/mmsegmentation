import os.path as osp
import numpy as np
from PIL import Image

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import fileio

@DATASETS.register_module()
class CropAndWeedDataset(BaseSegDataset):
    # Define METAINFO dict using classes and palette from DATASETS dict
    METAINFO = {
        'classes': [
            'Soil', 'Maize', 'Maize two-leaf stage', 'Maize four-leaf stage', 'Maize six-leaf stage',
            'Maize eight-leaf stage', 'Maize max', 'Sugar beet', 'Sugar beet two-leaf stage',
            'Sugar beet four-leaf stage', 'Sugar beet six-leaf stage', 'Sugar beet eight-leaf stage',
            'Sugar beet Max', 'Pea', 'Courgette', 'Pumpkins', 'Radish', 'Asparagus', 'Potato',
            'Flat leaf parsley', 'Curly leaf parsley', 'Cowslip', 'Poppy', 'Hemp', 'Sunflower',
            'Sage', 'Common bean', 'Faba bean', 'Clover', 'Hybrid goosefoot', 'Black-bindweed',
            'Cockspur grass', 'Red-root amaranth', 'White goosefoot', 'Thorn apple', 'Potato weed',
            'German chamomile', 'Saltbush', 'Creeping thistle', 'Field milk thistle', 'Purslane',
            'Black nightshade', 'Mercuries', 'Spurge', 'Pale persicaria', 'Geraniums', 'Cleavers',
            'Whitetop', 'Meadow-grass', 'Frosted orach', 'Black horehound', 'Shepherds purse',
            'Field bindweed', 'Common mugwort', 'Hedge mustard', 'Groundsel', 'Speedwell',
            'Broadleaf plantain', 'White ball-mustard', 'Peppermint', 'Field pennycress',
            'Corn spurry', 'Purple crabgrass', 'Common fumitory', 'Ivy-leaved speedwell',
            'Annual meadow grass', 'Redshank', 'Common hemp-nettle', 'Rough meadow-grass',
            'Green bristlegrass', 'Small geranium', 'Cornflower', 'Common corn-cockle',
            'Creeping crowfoot', 'Wall barley', 'Annual fescue', 'Purple dead-nettle',
            'Ribwort plantain', 'Pineappleweed', 'Common chickweed', 'Hedge mustard', 'Soft brome',
            'Wild pansy', 'Yellow rocket', 'Common wild oat', 'Red poppy', 'Rye brome', 'Knotgrass',
            'Prickly lettuce', 'Copse-bindweed', 'Manyseeds', 'Common buckwheat', 'Chives',
            'Garlic', 'Soybean', 'Wild carrot', 'Field mustard', 'Giant fennel',
            'Common horsetail', 'Common dandelion', 'Vegetation'
        ],
        'palette': [
            (0, 0, 0), (255, 0, 0), (234, 0, 0), (212, 0, 0), (191, 0, 0), (170, 0, 0), (149, 0, 0),
            (255, 85, 0), (234, 78, 0), (212, 71, 0), (191, 64, 0), (170, 57, 0), (149, 50, 0),
            (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
            (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85), (255, 188, 178),
            (255, 207, 178), (255, 226, 178), (255, 245, 178), (245, 255, 178), (226, 255, 178),
            (207, 255, 178), (188, 255, 178), (178, 255, 188), (178, 255, 207), (178, 255, 226),
            (178, 255, 245), (178, 245, 255), (178, 226, 255), (178, 207, 255), (178, 188, 255),
            (188, 178, 255), (207, 178, 255), (226, 178, 255), (245, 178, 255), (255, 178, 245),
            (255, 178, 226), (255, 178, 207), (255, 178, 188), (255, 194, 178), (255, 213, 178),
            (255, 219, 178), (255, 232, 178), (255, 238, 178), (255, 251, 178), (255, 212, 0),
            (239, 255, 178), (233, 255, 178), (220, 255, 178), (214, 255, 178), (201, 255, 178),
            (195, 255, 178), (182, 255, 178), (178, 255, 194), (178, 255, 200), (178, 255, 213),
            (178, 255, 220), (178, 255, 232), (178, 255, 238), (178, 255, 251), (178, 239, 255),
            (178, 233, 255), (178, 220, 255), (178, 214, 255), (178, 201, 255), (178, 195, 255),
            (178, 182, 255), (194, 178, 255), (200, 178, 255), (213, 178, 255), (219, 178, 255),
            (232, 178, 255), (238, 178, 255), (251, 178, 255), (255, 178, 239), (255, 178, 233),
            (255, 178, 220), (255, 178, 214), (212, 255, 0), (127, 255, 0), (42, 255, 0),
            (244, 255, 0), (159, 255, 0), (74, 255, 0), (10, 255, 0), (202, 255, 0), (128, 128, 128)
        ]
    }

    def __init__(self, variant='CropAndWeed', **kwargs):
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

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        return self.get_data_info(idx)