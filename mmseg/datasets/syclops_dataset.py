from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import fileio
import numpy as np
import os.path as osp
import os
import random
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

@DATASETS.register_module()
class SyclopsDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crop', 'weed'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
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

@DATASETS.register_module()
class SyclopsDatasetCS(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crop', 'weed', 'other'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]]
    )

    def __init__(self, file_list='./mmseg_output/eccv_results/Deeplabv3Plus_r50_syn_ohem_loss_dilated_masks/merged_all.txt', 
                subset_fraction=1.0, 
                random_seed=None,
                **kwargs):
        """Initialize the dataset.
        
        Args:
            file_list (str): Path to text file containing list of PNG files.
            subset_fraction (float): Fraction of dataset to use (between 0 and 1)
            random_seed (int, optional): Random seed for reproducibility
            **kwargs: Additional arguments passed to BaseSegDataset
        """
        self.subset_fraction = max(0.0, min(1.0, subset_fraction))  # Clip between 0 and 1
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
        
        # Read the file list from the provided path
        self.sample_list = []
        try:
            with open(file_list, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        self.sample_list.append(line)
            print(f'Loaded {len(self.sample_list)} samples from {file_list}')
        except FileNotFoundError:
            print(f'Warning: File list not found at {file_list}. Using empty sample list.')
            self.sample_list = []
        except Exception as e:
            print(f'Error reading file list from {file_list}: {e}')
            self.sample_list = [] 
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        
    def load_data_list(self):
        """Load annotation from sample list and optionally subset the data.
        Returns:
            list[dict]: Selected data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        
        # First collect all data from sample list
        for img_file in self.sample_list:
            # Remove .png extension if present and add it back consistently
            img_name = img_file.replace(self.img_suffix, '') + self.img_suffix
            
            data_info = dict(
                img_path=osp.join(img_dir, img_name) if img_dir else img_name,
                seg_fields=[],
                sample_idx=len(data_list)
            )
            if ann_dir is not None:
                seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
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


@DATASETS.register_module()
class SyclopsDatasetDilatedWeedInstances(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crop', 'weed'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    )

    def __init__(self,
                 instance_map_path,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 instance_map_suffix='.npz',
                 dilate_kernel_size=5,
                 dilate_iterations=1,
                 dump_debug_samples=True,
                 debug_dump_max_samples=12,
                 debug_dump_dir='./mmseg_output/eccv_results/Deeplabv3Plus_r50_syn_ohem_loss_dilated_masks',
                 debug_overlay_alpha=0.5,
                 **kwargs):
        if cv2 is None:
            raise ImportError('cv2 is required for SyclopsDatasetDilatedWeedInstances.')

        self.instance_map_path = instance_map_path
        self.instance_map_suffix = instance_map_suffix
        self.dilate_kernel_size = max(1, int(dilate_kernel_size))
        self.dilate_iterations = max(1, int(dilate_iterations))
        self.dump_debug_samples = bool(dump_debug_samples)
        self.debug_dump_max_samples = max(0, int(debug_dump_max_samples))
        self.debug_dump_dir = debug_dump_dir
        self.debug_overlay_alpha = float(max(0.0, min(1.0, debug_overlay_alpha)))
        self._debug_dump_count = 0

        if self.dilate_kernel_size % 2 == 0:
            self.dilate_kernel_size += 1

        self._kernel = np.ones(
            (self.dilate_kernel_size, self.dilate_kernel_size), dtype=np.uint8)

        if self.dump_debug_samples and self.debug_dump_max_samples > 0:
            os.makedirs(self.debug_dump_dir, exist_ok=True)

        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)

    def load_data_list(self):
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)

        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True):
            stem = img[:-len(self.img_suffix)]
            data_info = dict(
                img_path=osp.join(img_dir, img),
                seg_fields=[])
            if ann_dir is not None:
                data_info['seg_map_path'] = osp.join(ann_dir, stem + self.seg_map_suffix)
            data_info['instance_map_path'] = osp.join(
                self.instance_map_path, stem + self.instance_map_suffix)
            data_info['label_map'] = None
            data_info['reduce_zero_label'] = False
            data_info['seg_fields'] = []
            data_list.append(data_info)

        return sorted(data_list, key=lambda x: x['img_path'])

    def _load_npz_array(self, npz_path):
        with np.load(npz_path) as npz_data:
            if 'array' in npz_data:
                array = npz_data['array'].copy()
            else:
                first_key = list(npz_data.keys())[0]
                array = npz_data[first_key].copy()
        return np.asarray(array).squeeze()

    def _dilate_weed_by_instance(self, seg_map, instance_map):
        output = seg_map.copy()
        weed_sem = seg_map == 2
        weed_instance_ids = np.unique(instance_map[weed_sem])
        weed_instance_ids = weed_instance_ids[weed_instance_ids > 0]

        for instance_id in weed_instance_ids:
            instance_area = instance_map == instance_id
            seed = np.logical_and(instance_area, weed_sem).astype(np.uint8)
            if seed.sum() == 0:
                continue

            dilated = cv2.dilate(seed, self._kernel, iterations=self.dilate_iterations) > 0
            dilated = np.logical_and(dilated, instance_area)
            output[dilated] = 2

        return output.astype(np.uint8)

    def _mask_to_color(self, seg_map):
        color = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        color[seg_map == 0] = np.array([0, 0, 0], dtype=np.uint8)
        color[seg_map == 1] = np.array([0, 255, 0], dtype=np.uint8)
        color[seg_map == 2] = np.array([255, 0, 0], dtype=np.uint8)
        return color

    def _dump_debug(self, img_path, seg_map_dilated):
        if not self.dump_debug_samples:
            return
        if self._debug_dump_count >= self.debug_dump_max_samples:
            return

        base_name = osp.splitext(osp.basename(img_path))[0]
        out_prefix = f'{self._debug_dump_count:03d}_{base_name}'
        color_mask = self._mask_to_color(seg_map_dilated)

        mask_out = osp.join(self.debug_dump_dir, out_prefix + '_dilated_mask.png')
        Image.fromarray(color_mask).save(mask_out)

        if osp.exists(img_path):
            rgb = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            if rgb.shape[:2] == seg_map_dilated.shape:
                overlay = (
                    (1.0 - self.debug_overlay_alpha) * rgb
                    + self.debug_overlay_alpha * color_mask
                ).clip(0, 255).astype(np.uint8)
                overlay_out = osp.join(self.debug_dump_dir, out_prefix + '_overlay.png')
                Image.fromarray(overlay).save(overlay_out)

        self._debug_dump_count += 1

    def load_annotations(self, img_path, seg_map_path, instance_map_path):
        img_info = dict(filename=img_path, seg_fields=[])
        seg_map = self._load_npz_array(seg_map_path).astype(np.uint8)
        instance_map = self._load_npz_array(instance_map_path).astype(np.int64)

        if seg_map.ndim != 2:
            raise ValueError(f'Segmentation map should be 2D, got shape {seg_map.shape}')
        if instance_map.ndim != 2:
            raise ValueError(f'Instance map should be 2D, got shape {instance_map.shape}')
        if seg_map.shape != instance_map.shape:
            raise ValueError(
                f'Shape mismatch: semantic={seg_map.shape}, instance={instance_map.shape}')

        seg_map_dilated = self._dilate_weed_by_instance(seg_map, instance_map)
        self._dump_debug(img_path, seg_map_dilated)

        img_info['gt_seg_map'] = seg_map_dilated
        img_info['seg_fields'].append('gt_seg_map')
        return img_info

    def get_ann_info(self, idx):
        data = self.data_list[idx]
        return self.load_annotations(
            data['img_path'], data['seg_map_path'], data['instance_map_path'])