import os
import os.path as osp
from pathlib import Path
import random
from collections import deque
import numpy as np
import torch
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import fileio
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
# from tools.minimal_inference import ImageProcessor

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
                dir_path=ann_dir,
                list_dir=False,
                suffix=self.seg_map_suffix,
                recursive=True):
            img_file = img[:-4]+self.img_suffix
            data_info = dict(img_path=osp.join(img_dir, img_file))
            if ann_dir is not None:
                seg_map = img[:-4] + self.seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = None
            data_info['reduce_zero_label'] = False
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
    
@DATASETS.register_module()
class PhenobenchDatasetRegionBased(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crop', 'weed'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    )

    def __init__(self,
                 active_mask_dir='semantics_active_mask',
                 active_indicator_dir='semantics_active_indicator',
                 **kwargs):
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
            data_info['active_indicator_path'] = osp.join(active_indicator_dir, img[:-4]+'.pth')
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
class PhenobenchDatasetCopyPasteWeed(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'crop', 'weed'),
        palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]])

    def __init__(self,
                 syclops_img_path,
                 syclops_seg_map_path,
                 syclops_img_suffix='.png',
                 syclops_seg_map_suffix='.png',
                 syclops_pool_num_images=500,
                 show_pool_progress=True,
                 num_weeds_range=(1, 3),
                 min_weed_area=20,
                 max_placement_trials=50,
                 dump_debug_samples=True,
                 debug_dump_max_samples=12,
                 random_seed=None,
                 **kwargs):
        self.syclops_img_path = syclops_img_path
        self.syclops_seg_map_path = syclops_seg_map_path
        self.syclops_img_suffix = syclops_img_suffix
        self.syclops_seg_map_suffix = syclops_seg_map_suffix
        self.syclops_pool_num_images = (
            None if syclops_pool_num_images is None else int(syclops_pool_num_images))
        self.show_pool_progress = bool(show_pool_progress)
        self.min_weed_area = max(1, int(min_weed_area))
        self.max_placement_trials = max(1, int(max_placement_trials))
        self.dump_debug_samples = bool(dump_debug_samples)
        self.debug_dump_max_samples = max(0, int(debug_dump_max_samples))
        self._rng = random.Random(random_seed)
        self._src_img_cache = {}
        self._debug_dump_count = 0
        self._debug_dump_root = '/netscratch/naeem/mmseg_output/eccv_results/Deeplabv3Plus_r50_phenobench_with_syn_copy_pasting_focal_loss_weighted/copypaste'
        self._debug_rgb_dir = osp.join(self._debug_dump_root, 'rgb')
        self._debug_mask_dir = osp.join(self._debug_dump_root, 'mask')

        if self.dump_debug_samples and self.debug_dump_max_samples > 0:
            os.makedirs(self._debug_rgb_dir, exist_ok=True)
            os.makedirs(self._debug_mask_dir, exist_ok=True)

        if len(num_weeds_range) != 2:
            raise ValueError('num_weeds_range must be a 2-item tuple/list (min, max).')
        min_weeds, max_weeds = int(num_weeds_range[0]), int(num_weeds_range[1])
        if min_weeds < 0:
            min_weeds = 0
        if max_weeds < min_weeds:
            max_weeds = min_weeds
        self.num_weeds_range = (min_weeds, max_weeds)

        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

        self._weed_pool = self._build_weed_pool()
        if len(self._weed_pool) == 0:
            print('Warning: no weed instances found in Syclops source. Copy-paste is disabled.')

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
                data_info['seg_map_path'] = osp.join(ann_dir, img)
            data_info['label_map'] = None
            data_info['reduce_zero_label'] = False
            data_info['seg_fields'] = []
            data_list.append(data_info)

        return sorted(data_list, key=lambda x: x['img_path'])

    def get_ann_info(self, idx):
        return self.get_data_info(idx)

    def prepare_data(self, idx):
        """Load image/mask, apply weed copy-paste, then run pipeline.

        Loader transforms (`LoadImageFromFile`, `LoadImageFromNDArray`,
        `LoadAnnotations`) are skipped because data is already in memory.
        """
        data_info = self.get_data_info(idx)
        img = self._read_rgb_image(data_info['img_path'])

        results = dict(data_info)
        results['img'] = img
        results['img_path'] = data_info['img_path']
        results['ori_shape'] = img.shape[:2]
        results['img_shape'] = img.shape[:2]
        results['seg_fields'] = []

        seg_map_path = data_info.get('seg_map_path', None)
        if seg_map_path is not None and osp.exists(seg_map_path):
            gt_seg_map = self._read_seg_map(seg_map_path)
            if gt_seg_map.shape != img.shape[:2]:
                raise ValueError(
                    f'Image/mask shape mismatch for {data_info["img_path"]}: '
                    f'img={img.shape[:2]}, mask={gt_seg_map.shape}')
            img, gt_seg_map = self._apply_copy_paste(img, gt_seg_map)
            self._maybe_dump_debug_sample(data_info['img_path'], idx, img, gt_seg_map)
            results['img'] = img
            results['gt_seg_map'] = gt_seg_map
            results['seg_fields'].append('gt_seg_map')

        for transform in self.pipeline.transforms:
            transform_name = transform.__class__.__name__
            if transform_name in {'LoadImageFromFile', 'LoadImageFromNDArray', 'LoadAnnotations'}:
                continue
            results = transform(results)
            if results is None:
                return None
        return results

    def _sample_num_weeds(self):
        min_weeds, max_weeds = self.num_weeds_range
        return self._rng.randint(min_weeds, max_weeds)

    def _read_rgb_image(self, img_path):
        return np.array(Image.open(img_path).convert('RGB'))

    def _read_seg_map(self, seg_map_path):
        if seg_map_path.endswith('.npz'):
            npz_data = np.load(seg_map_path)
            if 'array' in npz_data:
                seg_map = npz_data['array']
            else:
                first_key = list(npz_data.keys())[0]
                seg_map = npz_data[first_key]
        else:
            seg_map = np.array(Image.open(seg_map_path))

        seg_map = np.asarray(seg_map).squeeze().astype(np.uint8)
        seg_map[seg_map == 3] = 1
        seg_map[seg_map == 4] = 2
        return seg_map

    def _build_weed_pool(self):
        weed_pool = []
        suffix_len = len(self.syclops_img_suffix)

        img_rel_list = list(fileio.list_dir_or_file(
            dir_path=self.syclops_img_path,
            list_dir=False,
            suffix=self.syclops_img_suffix,
            recursive=True))
        img_rel_list = sorted(img_rel_list)

        if self.syclops_pool_num_images is not None:
            if self.syclops_pool_num_images <= 0:
                return weed_pool
            if len(img_rel_list) > self.syclops_pool_num_images:
                img_rel_list = img_rel_list[:self.syclops_pool_num_images]

        iterator = img_rel_list
        if self.show_pool_progress:
            iterator = tqdm(
                img_rel_list,
                desc='Building weed pool',
                unit='img',
                leave=False)

        for img_rel in iterator:
            img_path = osp.join(self.syclops_img_path, img_rel)
            seg_rel = img_rel[:-suffix_len] + self.syclops_seg_map_suffix
            seg_path = osp.join(self.syclops_seg_map_path, seg_rel)
            if not osp.exists(seg_path):
                continue

            seg_map = self._read_seg_map(seg_path)
            if seg_map.ndim != 2:
                continue

            weed_binary = seg_map == 2
            components = self._extract_connected_components(weed_binary)
            for comp in components:
                if int(comp['mask'].sum()) < self.min_weed_area:
                    continue
                weed_pool.append(
                    dict(
                        src_img_path=img_path,
                        y1=comp['y1'],
                        y2=comp['y2'],
                        x1=comp['x1'],
                        x2=comp['x2'],
                        mask=comp['mask']))

        return weed_pool

    def _extract_connected_components(self, binary_mask):
        h, w = binary_mask.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []
        ys, xs = np.where(binary_mask)

        for sy, sx in zip(ys, xs):
            if visited[sy, sx]:
                continue

            queue = deque([(int(sy), int(sx))])
            visited[sy, sx] = True
            coords = []

            while queue:
                y, x = queue.popleft()
                coords.append((y, x))

                if y > 0 and binary_mask[y - 1, x] and not visited[y - 1, x]:
                    visited[y - 1, x] = True
                    queue.append((y - 1, x))
                if y + 1 < h and binary_mask[y + 1, x] and not visited[y + 1, x]:
                    visited[y + 1, x] = True
                    queue.append((y + 1, x))
                if x > 0 and binary_mask[y, x - 1] and not visited[y, x - 1]:
                    visited[y, x - 1] = True
                    queue.append((y, x - 1))
                if x + 1 < w and binary_mask[y, x + 1] and not visited[y, x + 1]:
                    visited[y, x + 1] = True
                    queue.append((y, x + 1))

            if not coords:
                continue

            y_coords = np.array([p[0] for p in coords], dtype=np.int32)
            x_coords = np.array([p[1] for p in coords], dtype=np.int32)
            y1, y2 = int(y_coords.min()), int(y_coords.max()) + 1
            x1, x2 = int(x_coords.min()), int(x_coords.max()) + 1

            comp_mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
            comp_mask[y_coords - y1, x_coords - x1] = True
            components.append(dict(y1=y1, y2=y2, x1=x1, x2=x2, mask=comp_mask))

        return components

    def _get_src_image(self, src_img_path):
        src_img = self._src_img_cache.get(src_img_path, None)
        if src_img is None:
            src_img = self._read_rgb_image(src_img_path)
            self._src_img_cache[src_img_path] = src_img
        return src_img

    def _apply_copy_paste(self, target_img, target_mask):
        if len(self._weed_pool) == 0:
            return target_img, target_mask

        num_to_insert = self._sample_num_weeds()
        if num_to_insert <= 0:
            return target_img, target_mask

        out_img = target_img.copy()
        out_mask = target_mask.copy()
        h, w = out_mask.shape

        for _ in range(num_to_insert):
            weed_item = self._rng.choice(self._weed_pool)
            weed_mask = weed_item['mask']
            weed_h, weed_w = weed_mask.shape

            if weed_h > h or weed_w > w:
                continue

            src_img = self._get_src_image(weed_item['src_img_path'])
            weed_rgb = src_img[weed_item['y1']:weed_item['y2'], weed_item['x1']:weed_item['x2']]

            if weed_rgb.shape[:2] != weed_mask.shape:
                continue

            for _ in range(self.max_placement_trials):
                top = self._rng.randint(0, h - weed_h)
                left = self._rng.randint(0, w - weed_w)

                target_crop = out_mask[top:top + weed_h, left:left + weed_w]
                if np.any(target_crop[weed_mask] != 0):
                    continue

                out_crop = out_img[top:top + weed_h, left:left + weed_w]
                out_crop[weed_mask] = weed_rgb[weed_mask]
                target_crop[weed_mask] = 2
                break

        return out_img, out_mask

    def _mask_to_color(self, mask):
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mask[mask == 0] = np.array([0, 0, 0], dtype=np.uint8)
        color_mask[mask == 1] = np.array([0, 255, 0], dtype=np.uint8)
        color_mask[mask == 2] = np.array([255, 0, 0], dtype=np.uint8)
        return color_mask

    def _maybe_dump_debug_sample(self, img_path, idx, img, mask):
        if not self.dump_debug_samples:
            return
        if self._debug_dump_count >= self.debug_dump_max_samples:
            return

        base_name = osp.splitext(osp.basename(img_path))[0]
        dump_name = f'{self._debug_dump_count:03d}_idx{idx}_{base_name}_pid{os.getpid()}'
        rgb_out_path = osp.join(self._debug_rgb_dir, dump_name + '.png')
        mask_out_path = osp.join(self._debug_mask_dir, dump_name + '.png')

        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(rgb_out_path)
        color_mask = self._mask_to_color(np.asarray(mask, dtype=np.uint8))
        Image.fromarray(color_mask).save(mask_out_path)

        self._debug_dump_count += 1
    
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