# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import OrderedDict
from datetime import datetime
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
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
        instance_map_subdirs (Sequence[str]): Candidate subdir names used when
            inferring instance map path from semantic mask path. Earlier
            entries have higher priority.
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
                 instance_map_subdirs: Sequence[str] = ('plant_instances',
                                                       'instance_segmentation'),
                 resize_instance_to_gt: bool = True,
                 vis_output_dir: Optional[str] = None,
                 vis_area_bins: Optional[Sequence[str]] = None,
                 vis_class: str = 'weed',
                 show_pred_for_detected_only: bool = True,
                 vis_gt_alpha: float = 0.45,
                 vis_pred_alpha: float = 0.35,
                 vis_per_eval_subdir: bool = True,
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
        self.instance_map_subdirs = tuple(instance_map_subdirs)
        self.resize_instance_to_gt = bool(resize_instance_to_gt)
        self.area_bins = tuple(area_bins) if area_bins is not None else self.default_area_bins
        self.vis_output_dir = vis_output_dir
        self.vis_area_bins = set(vis_area_bins) if vis_area_bins is not None else None
        if vis_class not in {'crop', 'weed', 'all'}:
            raise ValueError(f"vis_class must be one of ['crop', 'weed', 'all'], got {vis_class}.")
        self.vis_class = vis_class
        self.show_pred_for_detected_only = bool(show_pred_for_detected_only)
        self.vis_gt_alpha = float(max(0.0, min(1.0, vis_gt_alpha)))
        self.vis_pred_alpha = float(max(0.0, min(1.0, vis_pred_alpha)))
        self.vis_per_eval_subdir = bool(vis_per_eval_subdir)
        self._vis_sample_index = 0
        self._vis_initialized = False
        self._vis_eval_output_dir = None
        self._vis_count_by_key = OrderedDict()
        self._vis_manifest = []

        if self.vis_output_dir is not None:
            mkdir_or_exist(self.vis_output_dir)

        self._label_to_name = {
            self.crop_label: 'crop',
            self.weed_label: 'weed',
        }

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples."""
        if self.vis_output_dir is not None and not self._vis_initialized:
            self._prepare_vis_output_dir()
            self._vis_initialized = True

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

            instance_records = self._evaluate_single_records(pred_label, gt_label, instance_map)
            sample_stats = [
                dict(
                    class_name=record['class_name'],
                    area=record['area'],
                    detected=record['detected'])
                for record in instance_records
            ]
            # Wrap the list of instances for this image in a single dict
            # so that mmengine's collect_results counts it as 1 sample.
            self.results.append({'instances': sample_stats})

            if self.vis_output_dir is not None:
                img_path = self._resolve_img_path(data_sample)
                if img_path is not None and osp.exists(img_path):
                    rgb_image = self._load_rgb_image(img_path)
                    if rgb_image.shape[:2] != gt_label.shape:
                        rgb_image = self._resize_rgb_nearest(rgb_image, gt_label.shape)
                    self._dump_visualizations(
                        img_path=img_path,
                        rgb_image=rgb_image,
                        pred_label=pred_label,
                        instance_records=instance_records)
        # print('hold')

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

        # Flatten the per-image instance lists back into a single list
        flat_results = []
        for res in results:
            flat_results.extend(res.get('instances', []))

        for item in flat_results:
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

        self._log_visualization_audit(logger=logger, results=flat_results)

        self._vis_initialized = False
        self._vis_sample_index = 0
        self._vis_count_by_key = OrderedDict()
        self._vis_manifest = []
        return rounded_metrics

    def _prepare_vis_output_dir(self) -> None:
        if self.vis_output_dir is None:
            self._vis_eval_output_dir = None
            return

        if self.vis_per_eval_subdir:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self._vis_eval_output_dir = osp.join(
                self.vis_output_dir, f'eval_{stamp}_pid{os.getpid()}')
        else:
            self._vis_eval_output_dir = self.vis_output_dir

        mkdir_or_exist(self._vis_eval_output_dir)
        logger: MMLogger = MMLogger.get_current_instance()
        print_log(f'instance detection visualizations will be saved to: {self._vis_eval_output_dir}',
                  logger=logger)

    def _evaluate_single(self, pred_label: np.ndarray, gt_label: np.ndarray,
                         instance_map: np.ndarray) -> List[dict]:
        records = self._evaluate_single_records(pred_label, gt_label, instance_map)
        return [
            dict(class_name=record['class_name'], area=record['area'], detected=record['detected'])
            for record in records
        ]

    def _evaluate_single_records(self, pred_label: np.ndarray, gt_label: np.ndarray,
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

            bin_name = self._area_to_bin(area)

            sample_stats.append(
                dict(
                    instance_id=int(instance_id),
                    class_name=class_name,
                    class_label=class_label,
                    area=area,
                    detected=overlap >= self.overlap_thr,
                    overlap=overlap,
                    bin_name=bin_name,
                    inst_mask=inst_mask,
                ))

        return sample_stats

    def _resolve_img_path(self, data_sample: dict) -> Optional[str]:
        img_path = data_sample.get('img_path', None)
        if img_path is None and hasattr(data_sample, 'metainfo'):
            img_path = data_sample.metainfo.get('img_path', None)
        return img_path

    def _load_rgb_image(self, img_path: str) -> np.ndarray:
        return np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

    def _dump_visualizations(self,
                             img_path: str,
                             rgb_image: np.ndarray,
                             pred_label: np.ndarray,
                             instance_records: List[dict]) -> None:
        basename = osp.splitext(osp.basename(img_path))[0]

        for record in instance_records:
            if not self._should_visualize_record(record):
                continue

            class_name = record['class_name']
            detected = bool(record['detected'])
            bin_name = record['bin_name']
            status = 'detected' if detected else 'missed'

            vis_root = self._vis_eval_output_dir or self.vis_output_dir
            out_dir = osp.join(vis_root, bin_name, class_name, status)
            mkdir_or_exist(out_dir)

            vis_img = self._compose_instance_visual(
                rgb_image=rgb_image,
                pred_label=pred_label,
                record=record)

            out_name = (
                f'{self._vis_sample_index:08d}_{basename}_inst{record["instance_id"]}'
                f'_area{record["area"]}_{status}.png')
            out_path = osp.join(out_dir, out_name)
            Image.fromarray(vis_img).save(out_path)

            count_key = (bin_name, class_name, status)
            self._vis_count_by_key[count_key] = self._vis_count_by_key.get(count_key, 0) + 1
            self._vis_manifest.append(dict(
                path=out_path,
                bin_name=bin_name,
                class_name=class_name,
                status=status,
                area=int(record['area']),
                instance_id=int(record['instance_id']),
                overlap=float(record['overlap']),
            ))
            self._vis_sample_index += 1

    def _should_visualize_record(self, record: dict) -> bool:
        if self.vis_area_bins is not None and record['bin_name'] not in self.vis_area_bins:
            return False

        if self.vis_class != 'all' and record['class_name'] != self.vis_class:
            return False
        return True

    def _compose_instance_visual(self,
                                 rgb_image: np.ndarray,
                                 pred_label: np.ndarray,
                                 record: dict) -> np.ndarray:
        base = np.asarray(rgb_image, dtype=np.uint8).copy()
        inst_mask = record['inst_mask']

        if record['class_name'] == 'weed':
            gt_color = np.array([255, 0, 0], dtype=np.uint8)
        else:
            gt_color = np.array([0, 255, 0], dtype=np.uint8)
        pred_color = np.array([0, 0, 255], dtype=np.uint8)

        base[inst_mask] = (
            (1.0 - self.vis_gt_alpha) * base[inst_mask]
            + self.vis_gt_alpha * gt_color
        ).astype(np.uint8)

        draw_pred = (not self.show_pred_for_detected_only) or bool(record['detected'])
        if draw_pred:
            pred_mask_cls = pred_label == record['class_label']
            pred_on_instance = np.logical_and(pred_mask_cls, inst_mask)
            base[pred_on_instance] = (
                (1.0 - self.vis_pred_alpha) * base[pred_on_instance]
                + self.vis_pred_alpha * pred_color
            ).astype(np.uint8)

        ys, xs = np.where(inst_mask)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        pil_img = Image.fromarray(base)
        drawer = ImageDraw.Draw(pil_img)
        edge_color = (255, 255, 255)
        drawer.rectangle([(x1, y1), (x2, y2)], outline=edge_color, width=2)

        status = 'detected' if record['detected'] else 'missed'
        text = (
            f'{record["class_name"]} | area={record["area"]} px | '
            f'bin={record["bin_name"]} | {status} | ov={record["overlap"]:.3f}')
        text_x = x1
        text_y = max(0, y1 - 14)
        drawer.rectangle([(text_x, text_y), (min(text_x + 820, base.shape[1] - 1), text_y + 14)], fill=(0, 0, 0))
        drawer.text((text_x + 2, text_y), text, fill=(255, 255, 255))
        return np.array(pil_img, dtype=np.uint8)

    def _log_visualization_audit(self, logger: MMLogger, results: list) -> None:
        if self.vis_output_dir is None:
            return

        vis_root = self._vis_eval_output_dir or self.vis_output_dir

        if len(self._vis_manifest) > 0:
            manifest_path = osp.join(vis_root, 'manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump(self._vis_manifest, f, indent=2)
            print_log(f'instance detection visualization manifest: {manifest_path}', logger=logger)

        selected_total = 0
        selected_detected = 0
        for item in results:
            class_name = item['class_name']
            if self.vis_class != 'all' and class_name != self.vis_class:
                continue
            bin_name = self._area_to_bin(int(item['area']))
            if self.vis_area_bins is not None and bin_name not in self.vis_area_bins:
                continue
            selected_total += 1
            selected_detected += int(item['detected'])

        saved_detected = 0
        saved_missed = 0
        for (bin_name, class_name, status), count in self._vis_count_by_key.items():
            if self.vis_area_bins is not None and bin_name not in self.vis_area_bins:
                continue
            if self.vis_class != 'all' and class_name != self.vis_class:
                continue
            if status == 'detected':
                saved_detected += count
            else:
                saved_missed += count
        saved_total = saved_detected + saved_missed

        print_log(
            'instance detection vis audit: '
            f'expected_total={selected_total}, expected_detected={selected_detected}, '
            f'expected_missed={selected_total - selected_detected}, '
            f'saved_total={saved_total}, saved_detected={saved_detected}, saved_missed={saved_missed}',
            logger=logger)

        if saved_total != selected_total:
            print_log(
                'WARNING: visualization count mismatch detected for current eval run. '
                'Check manifest.json for exact saved entries.',
                logger=logger)

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
            candidates = []
            for subdir in self.instance_map_subdirs:
                candidates.append(
                    seg_no_ext.replace('/semantics/', f'/{subdir}/')
                    + self.instance_map_suffix)
            for candidate in candidates:
                if osp.exists(candidate):
                    return candidate
            return candidates[-1]

        candidates = []
        for suffix in self.auto_instance_suffixes:
            for subdir in self.instance_map_subdirs:
                candidates.append(
                    seg_no_ext.replace('/semantics/', f'/{subdir}/') + suffix)

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

    @staticmethod
    def _resize_rgb_nearest(rgb_image: np.ndarray,
                            target_shape: Tuple[int, int]) -> np.ndarray:
        dst_h, dst_w = target_shape
        pil_img = Image.fromarray(rgb_image)
        pil_img = pil_img.resize((dst_w, dst_h), resample=Image.BILINEAR)
        return np.array(pil_img, dtype=np.uint8)

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