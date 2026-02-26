# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable

from mmseg.registry import METRICS


@METRICS.register_module()
class InstanceDetectionMetric(BaseMetric):
    """Instance-level detection metric for semantic predictions.

    A GT instance is counted as detected when semantic prediction overlaps with
    that GT instance by at least ``overlap_thr``.

    Notes:
        - This metric is intended for synthetic datasets that provide
          instance maps (e.g. Syclops).
        - GT instance ids can be large int64 values.

    Args:
        overlap_thr (float): Detection threshold in [0, 1].
        overlap_mode (str): Overlap type. Options:
            - ``'gt'``: intersection / GT instance area
            - ``'iou'``: intersection / union
            Default: ``'gt'``.
        crop_label (int): Semantic class id for crop. Default: 1.
        weed_label (int): Semantic class id for weed. Default: 2.
        area_bins (Sequence[Tuple[int, Optional[int]]]): Instance-area bins in
            pixels. Right edge is exclusive. ``None`` means open ended.
        instance_map_path (str, optional): Optional root directory to resolve
            instance maps when ``instance_map_path`` is not present in sample
            metainfo.
        instance_map_suffix (str): Suffix for instance maps. Default: '.npz'.
        collect_device (str): 'cpu' or 'gpu'. Default: 'cpu'.
        prefix (str, optional): Metric name prefix.
    """

    default_area_bins = (
        (0, 40),
        (40, 80),
        (80, 100),
        (100, 200),
        (200, 500),
        (500, 1000),
        (1000, None),
    )

    def __init__(self,
                 overlap_thr: float = 0.5,
                 overlap_mode: str = 'gt',
                 crop_label: int = 1,
                 weed_label: int = 2,
                 area_bins: Optional[Sequence[Tuple[int, Optional[int]]]] = None,
                 instance_map_path: Optional[str] = None,
                 instance_map_suffix: Optional[str] = None,
                 auto_instance_suffixes: Sequence[str] = ('.npz', '.png', '.npy'),
                 resize_instance_to_gt: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.overlap_thr = float(overlap_thr)
        if not (0.0 <= self.overlap_thr <= 1.0):
            raise ValueError(f'overlap_thr must be in [0, 1], got {overlap_thr}.')

        if overlap_mode not in {'gt', 'iou'}:
            raise ValueError(
                f"overlap_mode must be one of ['gt', 'iou'], got {overlap_mode}.")
        self.overlap_mode = overlap_mode

        self.crop_label = int(crop_label)
        self.weed_label = int(weed_label)
        self.instance_map_path = instance_map_path
        self.instance_map_suffix = instance_map_suffix
        self.auto_instance_suffixes = tuple(auto_instance_suffixes)
        self.resize_instance_to_gt = bool(resize_instance_to_gt)
        self.area_bins = tuple(area_bins) if area_bins is not None else self.default_area_bins

        self._label_to_name = {
            self.crop_label: 'crop',
            self.weed_label: 'weed',
        }

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples."""
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze().cpu().numpy()
            gt_label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy()

            instance_map_file = self._resolve_instance_map_path(data_sample)
            instance_map = self._load_instance_map(instance_map_file)

            if instance_map.shape != gt_label.shape:
                if not self.resize_instance_to_gt:
                    raise ValueError(
                        f'Shape mismatch for {instance_map_file}: '
                        f'instance_map={instance_map.shape}, gt={gt_label.shape}')
                instance_map = self._resize_instance_map_nearest(
                    instance_map, target_shape=gt_label.shape)

            sample_stats = self._evaluate_single(pred_label, gt_label, instance_map)
            self.results.extend(sample_stats)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute metrics from processed results."""
        logger: MMLogger = MMLogger.get_current_instance()

        if len(results) == 0:
            logger.warning('No instances found for InstanceDetectionMetric.')
            return OrderedDict()

        class_totals = {'crop': 0, 'weed': 0}
        class_detected = {'crop': 0, 'weed': 0}
        bin_totals = OrderedDict((self._bin_name(b), 0) for b in self.area_bins)
        bin_detected = OrderedDict((self._bin_name(b), 0) for b in self.area_bins)
        cls_bin_totals = {
            'crop': OrderedDict((self._bin_name(b), 0) for b in self.area_bins),
            'weed': OrderedDict((self._bin_name(b), 0) for b in self.area_bins),
        }
        cls_bin_detected = {
            'crop': OrderedDict((self._bin_name(b), 0) for b in self.area_bins),
            'weed': OrderedDict((self._bin_name(b), 0) for b in self.area_bins),
        }

        for item in results:
            class_name = item['class_name']
            detected = int(item['detected'])
            area = int(item['area'])
            bin_name = self._area_to_bin(area)

            class_totals[class_name] += 1
            class_detected[class_name] += detected

            bin_totals[bin_name] += 1
            bin_detected[bin_name] += detected
            cls_bin_totals[class_name][bin_name] += 1
            cls_bin_detected[class_name][bin_name] += detected

        total_instances = class_totals['crop'] + class_totals['weed']
        total_detected = class_detected['crop'] + class_detected['weed']

        metrics = OrderedDict()
        metrics['inst_crop_total'] = float(class_totals['crop'])
        metrics['inst_crop_detected'] = float(class_detected['crop'])
        metrics['inst_weed_total'] = float(class_totals['weed'])
        metrics['inst_weed_detected'] = float(class_detected['weed'])
        metrics['inst_total'] = float(total_instances)
        metrics['inst_detected'] = float(total_detected)

        metrics['inst_crop_detAcc'] = self._safe_ratio(class_detected['crop'], class_totals['crop']) * 100.0
        metrics['inst_weed_detAcc'] = self._safe_ratio(class_detected['weed'], class_totals['weed']) * 100.0
        metrics['inst_overall_detAcc'] = self._safe_ratio(total_detected, total_instances) * 100.0

        for bin_name in bin_totals:
            metrics[f'inst_bin_{bin_name}_total'] = float(bin_totals[bin_name])
            metrics[f'inst_bin_{bin_name}_detected'] = float(bin_detected[bin_name])
            metrics[f'inst_bin_{bin_name}_detAcc'] = (
                self._safe_ratio(bin_detected[bin_name], bin_totals[bin_name]) * 100.0)

        for class_name in ('crop', 'weed'):
            for bin_name in cls_bin_totals[class_name]:
                metrics[f'inst_{class_name}_bin_{bin_name}_detAcc'] = (
                    self._safe_ratio(
                        cls_bin_detected[class_name][bin_name],
                        cls_bin_totals[class_name][bin_name]) * 100.0)

        self._log_summary_tables(
            logger,
            class_totals=class_totals,
            class_detected=class_detected,
            bin_totals=bin_totals,
            bin_detected=bin_detected,
            cls_bin_totals=cls_bin_totals,
            cls_bin_detected=cls_bin_detected)

        rounded_metrics = OrderedDict()
        for key, value in metrics.items():
            if key.endswith('_detAcc'):
                rounded_metrics[key] = round(value, 2)
            else:
                rounded_metrics[key] = value
        return rounded_metrics

    def _evaluate_single(self, pred_label: np.ndarray, gt_label: np.ndarray,
                         instance_map: np.ndarray) -> List[dict]:
        sample_stats = []
        instance_ids = np.unique(instance_map)
        instance_ids = instance_ids[instance_ids > 0]

        for instance_id in instance_ids:
            inst_mask = instance_map == instance_id
            area = int(inst_mask.sum())
            if area <= 0:
                continue

            class_name, class_label = self._instance_class(inst_mask, gt_label)
            if class_name is None:
                continue

            pred_mask_cls = pred_label == class_label
            inter = int(np.logical_and(inst_mask, pred_mask_cls).sum())

            if self.overlap_mode == 'gt':
                overlap = inter / float(area)
            else:
                union = int(np.logical_or(inst_mask, pred_mask_cls).sum())
                overlap = inter / float(union) if union > 0 else 0.0

            sample_stats.append(
                dict(
                    class_name=class_name,
                    area=area,
                    detected=overlap >= self.overlap_thr,
                ))

        return sample_stats

    def _instance_class(self, inst_mask: np.ndarray,
                        gt_label: np.ndarray) -> Tuple[Optional[str], Optional[int]]:
        labels = gt_label[inst_mask]
        labels = labels[np.logical_or(labels == self.crop_label,
                                      labels == self.weed_label)]
        if labels.size == 0:
            return None, None

        uniq, counts = np.unique(labels, return_counts=True)
        class_label = int(uniq[np.argmax(counts)])
        class_name = self._label_to_name.get(class_label, None)
        return class_name, class_label

    def _resolve_instance_map_path(self, data_sample: dict) -> str:
        if 'instance_map_path' in data_sample:
            return data_sample['instance_map_path']

        seg_map_path = data_sample.get('seg_map_path', None)
        if seg_map_path is None and hasattr(data_sample, 'metainfo'):
            seg_map_path = data_sample.metainfo.get('seg_map_path', None)

        inst_map_path = data_sample.get('instance_map_path', None)
        if inst_map_path is None and hasattr(data_sample, 'metainfo'):
            inst_map_path = data_sample.metainfo.get('instance_map_path', None)
        if inst_map_path is not None:
            return inst_map_path

        if self.instance_map_path is not None:
            img_path = data_sample.get('img_path', None)
            if img_path is None and hasattr(data_sample, 'metainfo'):
                img_path = data_sample.metainfo.get('img_path', None)
            if img_path is not None:
                stem = osp.splitext(osp.basename(img_path))[0]
                if self.instance_map_suffix is not None:
                    return osp.join(self.instance_map_path, stem + self.instance_map_suffix)
                for suffix in self.auto_instance_suffixes:
                    candidate = osp.join(self.instance_map_path, stem + suffix)
                    if osp.exists(candidate):
                        return candidate

        if seg_map_path is not None:
            guessed_file = self._guess_from_seg_map_path(seg_map_path)
            if guessed_file is not None:
                return guessed_file

        raise KeyError(
            'Cannot resolve instance_map_path. Add `instance_map_path` into '
            'PackSegInputs meta_keys or set metric.instance_map_path.')

    def _load_instance_map(self, instance_map_file: str) -> np.ndarray:
        if not osp.exists(instance_map_file):
            raise FileNotFoundError(f'Instance map file not found: {instance_map_file}')

        if instance_map_file.endswith('.npz'):
            with np.load(instance_map_file) as npz_data:
                if 'array' in npz_data:
                    instance_map = npz_data['array']
                else:
                    first_key = list(npz_data.keys())[0]
                    instance_map = npz_data[first_key]
        elif instance_map_file.endswith('.npy'):
            instance_map = np.load(instance_map_file)
        elif instance_map_file.endswith(('.png', '.tif', '.tiff', '.bmp')):
            instance_map = np.array(Image.open(instance_map_file))
        else:
            raise ValueError(
                f'Unsupported instance map file type: {instance_map_file}. '
                'Supported suffixes are .npz, .npy, .png, .tif, .tiff, .bmp')

        instance_map = np.asarray(instance_map).squeeze().astype(np.int64)
        if instance_map.ndim != 2:
            raise ValueError(
                f'Instance map should be 2D, got shape {instance_map.shape} '
                f'for {instance_map_file}.')
        return instance_map

    def _guess_from_seg_map_path(self, seg_map_path: str) -> Optional[str]:
        seg_no_ext = osp.splitext(seg_map_path)[0]

        if self.instance_map_suffix is not None:
            candidates = [
                seg_no_ext.replace('/semantics/', '/instance_segmentation/') + self.instance_map_suffix,
                seg_no_ext.replace('/semantics/', '/plant_instances/') + self.instance_map_suffix,
            ]
            for candidate in candidates:
                if osp.exists(candidate):
                    return candidate
            return candidates[-1]

        candidates = []
        for suffix in self.auto_instance_suffixes:
            candidates.append(seg_no_ext.replace('/semantics/', '/instance_segmentation/') + suffix)
            candidates.append(seg_no_ext.replace('/semantics/', '/plant_instances/') + suffix)

        for candidate in candidates:
            if osp.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _resize_instance_map_nearest(instance_map: np.ndarray,
                                     target_shape: Tuple[int, int]) -> np.ndarray:
        src_h, src_w = instance_map.shape
        dst_h, dst_w = target_shape
        if src_h == dst_h and src_w == dst_w:
            return instance_map

        y_idx = np.floor(np.arange(dst_h) * (src_h / dst_h)).astype(np.int64)
        x_idx = np.floor(np.arange(dst_w) * (src_w / dst_w)).astype(np.int64)
        y_idx = np.clip(y_idx, 0, src_h - 1)
        x_idx = np.clip(x_idx, 0, src_w - 1)
        return instance_map[y_idx[:, None], x_idx[None, :]].astype(np.int64)

    def _area_to_bin(self, area: int) -> str:
        for area_bin in self.area_bins:
            left, right = area_bin
            if right is None:
                if area >= left:
                    return self._bin_name(area_bin)
            elif left <= area < right:
                return self._bin_name(area_bin)
        return self._bin_name(self.area_bins[-1])

    @staticmethod
    def _bin_name(area_bin: Tuple[int, Optional[int]]) -> str:
        left, right = area_bin
        if right is None:
            return f'{left}_inf'
        return f'{left}_{right}'

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return float('nan')
        return numerator / denominator

    def _log_summary_tables(self,
                            logger: MMLogger,
                            class_totals: Dict[str, int],
                            class_detected: Dict[str, int],
                            bin_totals: OrderedDict,
                            bin_detected: OrderedDict,
                            cls_bin_totals: Dict[str, OrderedDict],
                            cls_bin_detected: Dict[str, OrderedDict]) -> None:
        class_table = PrettyTable()
        class_table.field_names = ['Class', 'Detected', 'Total', 'DetAcc (%)']
        for class_name in ('crop', 'weed'):
            det = class_detected[class_name]
            tot = class_totals[class_name]
            acc = self._safe_ratio(det, tot) * 100.0
            class_table.add_row([class_name, det, tot, np.round(acc, 2)])

        total_det = class_detected['crop'] + class_detected['weed']
        total_tot = class_totals['crop'] + class_totals['weed']
        total_acc = self._safe_ratio(total_det, total_tot) * 100.0
        class_table.add_row(['overall', total_det, total_tot, np.round(total_acc, 2)])

        bin_table = PrettyTable()
        bin_table.field_names = ['Area Bin (px)', 'Detected', 'Total', 'DetAcc (%)']
        for bin_name in bin_totals:
            det = bin_detected[bin_name]
            tot = bin_totals[bin_name]
            acc = self._safe_ratio(det, tot) * 100.0
            bin_table.add_row([bin_name, det, tot, np.round(acc, 2)])

        class_bin_table = PrettyTable()
        class_bin_table.field_names = ['Class', 'Area Bin (px)', 'Detected', 'Total', 'DetAcc (%)']
        for class_name in ('crop', 'weed'):
            for bin_name in cls_bin_totals[class_name]:
                det = cls_bin_detected[class_name][bin_name]
                tot = cls_bin_totals[class_name][bin_name]
                acc = self._safe_ratio(det, tot) * 100.0
                class_bin_table.add_row([class_name, bin_name, det, tot, np.round(acc, 2)])

        print_log('instance detection class summary:', logger=logger)
        print_log('\n' + class_table.get_string(), logger=logger)
        print_log('instance detection area-bin summary:', logger=logger)
        print_log('\n' + bin_table.get_string(), logger=logger)
        print_log('instance detection class x area-bin summary:', logger=logger)
        print_log('\n' + class_bin_table.get_string(), logger=logger)