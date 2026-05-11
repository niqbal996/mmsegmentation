# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import OrderedDict
from datetime import datetime
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as sp_ndimage
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

        Precision uses a deduplicated GT-hit definition per threshold:
        - TP is the number of unique GT instances hit by at least one predicted
            connected component with GT-normalized overlap strictly greater than
            threshold (``Area(gt ∩ pred_island) / Area(gt) > thr``).
        - FP_pure is the number of predicted connected components that do not hit
            any GT instance above threshold.
        - Precision is ``TP / (TP + FP_pure)``.

    Notes:
        - This metric is intended for synthetic datasets that provide
          instance maps (e.g. Syclops).
        - GT instance ids can be large int64 values.

    Args:
        overlap_thr (float): Legacy primary detection threshold in [0, 1],
            still used for visualization decisions when
            ``overlap_thrs_pct`` is not set.
        overlap_thrs_pct (Sequence[int], optional): Overlap thresholds in
            percent used for metric reporting in one run, e.g. ``(5, 10, 15)``.
            Values must be in (0, 100].
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
        vis_save_weed_island (bool): Whether to save full-image weed island
            precision visualizations. Default: True.
        pred_weed_morph_kernel_size (int): Morphology kernel size applied to
            predicted weed mask before connected-components. Supported values:
            0 (disabled), 3, 5. Default: 0.
        pred_weed_morph_op (str): Morphology operation for predicted weed mask
            when ``pred_weed_morph_kernel_size > 0``. Options:
            ``'dilation'``, ``'closing'``, ``'erosion'``. Default: 'dilation'.
        vis_save_fp_cases (bool): Whether to save sampled FP-focused side-by-side
            visualizations for weed islands. Default: True.
        vis_fp_case_max (int): Max number of FP-focused visualization images to
            save per evaluation run. Default: 20.
        vis_bg_alpha (float): Background alpha used for GT/pred side-by-side
            visualizations. Default: 0.35.
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
                 overlap_thrs_pct: Optional[Sequence[int]] = None,
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
                 vis_small_weed_pair_max: int = 20,
                 vis_save_weed_island: bool = True,
                 pred_weed_morph_kernel_size: int = 0,
                 pred_weed_morph_op: str = 'dilation',
                 vis_save_fp_cases: bool = True,
                 vis_fp_case_max: int = 20,
                 vis_bg_alpha: float = 0.35,
                 pred_island_filter_enable: bool = False,
                 pred_island_min_area_by_class: Optional[Dict[str, int]] = None,
                 vis_per_eval_subdir: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.overlap_thr = float(overlap_thr)
        if not (0.0 <= self.overlap_thr <= 1.0):
            raise ValueError(f'overlap_thr must be in [0, 1], got {overlap_thr}.')

        if overlap_thrs_pct is None:
            # Keep backward compatibility when users explicitly set overlap_thr.
            if abs(self.overlap_thr - 0.5) > 1e-12:
                overlap_thrs_pct = (int(round(self.overlap_thr * 100.0)),)
            else:
                overlap_thrs_pct = (5, 10, 15)
        self.overlap_thrs_pct = tuple(sorted(set(int(x) for x in overlap_thrs_pct)))
        if len(self.overlap_thrs_pct) == 0:
            raise ValueError('overlap_thrs_pct must not be empty.')
        for thr_pct in self.overlap_thrs_pct:
            if not (0 < thr_pct <= 100):
                raise ValueError(
                    'each entry in overlap_thrs_pct must be in (0, 100], '
                    f'got {thr_pct}.')
        self._overlap_thrs = tuple(x / 100.0 for x in self.overlap_thrs_pct)
        self.primary_overlap_thr = self._overlap_thrs[0]
        self.primary_overlap_thr_pct = self.overlap_thrs_pct[0]

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
        self.vis_small_weed_pair_max = int(max(0, vis_small_weed_pair_max))
        self.vis_save_weed_island = bool(vis_save_weed_island)
        self.pred_weed_morph_kernel_size = int(pred_weed_morph_kernel_size)
        if self.pred_weed_morph_kernel_size not in (0, 3, 5):
            raise ValueError(
                'pred_weed_morph_kernel_size must be one of [0, 3, 5], '
                f'got {pred_weed_morph_kernel_size}.')
        if pred_weed_morph_op not in {'dilation', 'closing', 'erosion'}:
            raise ValueError(
                "pred_weed_morph_op must be one of ['dilation', 'closing', 'erosion'], "
                f'got {pred_weed_morph_op}.')
        self.pred_weed_morph_op = pred_weed_morph_op
        self.vis_save_fp_cases = bool(vis_save_fp_cases)
        self.vis_fp_case_max = int(max(0, vis_fp_case_max))
        self.vis_bg_alpha = float(max(0.0, min(1.0, vis_bg_alpha)))
        self.pred_island_filter_enable = bool(pred_island_filter_enable)
        self.pred_island_min_area_by_class = dict(pred_island_min_area_by_class or {})
        self.vis_per_eval_subdir = bool(vis_per_eval_subdir)
        self._vis_sample_index = 0
        self._vis_small_pair_count = 0
        self._vis_initialized = False
        self._vis_eval_output_dir = None
        self._vis_count_by_key = OrderedDict()
        self._vis_manifest = []
        self._vis_fp_case_count = 0
        self._last_pred_island_removed_by_class = {'crop': 0, 'weed': 0}

        if self.vis_output_dir is not None:
            mkdir_or_exist(self.vis_output_dir)

        self._label_to_name = {
            self.crop_label: 'crop',
            self.weed_label: 'weed',
        }
        self._size_buckets = ('le100', 'ge100')

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
            pred_component_records = self._evaluate_predicted_components(
                pred_label, gt_label, instance_map)
            confusion_records, confusion_counts = self._compute_instance_class_confusions(
                pred_label=pred_label,
                instance_records=instance_records)

            # Store compact per-sample counters to keep memory stable even
            # when predicted connected components are numerous.
            sample_summary = self._summarize_sample_records(
                instance_records=instance_records,
                pred_component_records=pred_component_records,
                confusion_counts=confusion_counts)
            self.results.append({'summary': sample_summary})

            if self.vis_output_dir is not None:
                img_path = self._resolve_img_path(data_sample)
                if img_path is None:
                    logger: MMLogger = MMLogger.get_current_instance()
                    print_log('weed island vis skipped: img_path not found in data_sample', logger=logger)
                elif not osp.exists(img_path):
                    logger: MMLogger = MMLogger.get_current_instance()
                    print_log(f'weed island vis skipped: img_path does not exist: {img_path}', logger=logger)
                if img_path is not None and osp.exists(img_path):
                    rgb_image = self._load_rgb_image(img_path)
                    if rgb_image.shape[:2] != gt_label.shape:
                        rgb_image = self._resize_rgb_nearest(rgb_image, gt_label.shape)
                    self._dump_visualizations(
                        img_path=img_path,
                        rgb_image=rgb_image,
                        pred_label=pred_label,
                        instance_records=instance_records)

                    if self.vis_save_weed_island:
                        # Full-image weed island precision visualization (one per sample).
                        vis_root = self._vis_eval_output_dir or self.vis_output_dir
                        vis_island_dir = osp.join(vis_root, 'weed_island_vis')
                        mkdir_or_exist(vis_island_dir)
                        basename = osp.splitext(osp.basename(img_path))[0]
                        island_vis = self._compose_weed_island_precision_visual(
                            rgb_image=rgb_image,
                            pred_label=pred_label,
                            gt_label=gt_label)
                        island_out_path = osp.join(vis_island_dir, f'{basename}_weed_islands.png')
                        Image.fromarray(island_vis).save(island_out_path)
                        logger: MMLogger = MMLogger.get_current_instance()
                        print_log(f'weed island vis saved: {island_out_path}', logger=logger)

                    if self.vis_save_fp_cases and self._vis_fp_case_count < self.vis_fp_case_max:
                        fp_case = self._compose_fp_case_visual(
                            rgb_image=rgb_image,
                            pred_label=pred_label,
                            gt_label=gt_label,
                            overlap_thr=self.primary_overlap_thr,
                            confusion_records=confusion_records)
                        if fp_case is not None:
                            vis_root = self._vis_eval_output_dir or self.vis_output_dir
                            fp_case_dir = osp.join(vis_root, 'fp_sampled_cases')
                            mkdir_or_exist(fp_case_dir)
                            basename = osp.splitext(osp.basename(img_path))[0]
                            out_path = osp.join(
                                fp_case_dir,
                                f'{self._vis_fp_case_count:04d}_{basename}_fp_case.png')
                            Image.fromarray(fp_case).save(out_path)
                            self._vis_fp_case_count += 1
                            logger: MMLogger = MMLogger.get_current_instance()
                            print_log(f'fp sampled case saved: {out_path}', logger=logger)
        

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute metrics from processed results."""
        logger: MMLogger = MMLogger.get_current_instance()

        if len(results) == 0:
            logger.warning('No instances found for InstanceDetectionMetric.')
            return OrderedDict()

        classes = ('crop', 'weed')
        sizes = self._size_buckets
        thrs_pct = self.overlap_thrs_pct

        recall_totals = {
            cls: {
                size: {'total': 0, 'detected': OrderedDict((t, 0) for t in thrs_pct)}
                for size in sizes
            }
            for cls in classes
        }
        precision_totals = {
            cls: {
                size: {
                    'total_pred': 0,
                    'tp': OrderedDict((t, 0) for t in thrs_pct),
                    'fp_pure': OrderedDict((t, 0) for t in thrs_pct),
                }
                for size in sizes
            }
            for cls in classes
        }

        # Aggregate all counters from per-sample summaries.
        vis_selected_total = 0
        vis_selected_detected = 0
        removed_pred_islands_total = {'crop': 0, 'weed': 0}
        confusion_totals = {'weed_as_crop': 0, 'crop_as_weed': 0}

        for res in results:
            summary = res.get('summary', None)
            if summary is None:
                logger.warning(
                    'InstanceDetectionMetric: legacy result format detected; '
                    'skipping this entry because multi-threshold compact '
                    'aggregation requires new summary format.')
                continue

            recall_summary = summary['recall']
            precision_summary = summary['precision']

            for class_name in classes:
                for size_name in sizes:
                    recall_totals[class_name][size_name]['total'] += int(
                        recall_summary[class_name][size_name]['total'])
                    precision_totals[class_name][size_name]['total_pred'] += int(
                        precision_summary[class_name][size_name]['total_pred'])
                    for thr_pct in thrs_pct:
                        recall_totals[class_name][size_name]['detected'][thr_pct] += int(
                            recall_summary[class_name][size_name]['detected'][thr_pct])
                        precision_totals[class_name][size_name]['tp'][thr_pct] += int(
                            precision_summary[class_name][size_name]['tp'][thr_pct])
                        precision_totals[class_name][size_name]['fp_pure'][thr_pct] += int(
                            precision_summary[class_name][size_name]['fp_pure'][thr_pct])

            vis_selected_total += int(summary.get('vis_selected_total', 0))
            vis_selected_detected += int(summary.get('vis_selected_detected', 0))
            confusion_counts = summary.get('confusion_counts', {})
            confusion_totals['weed_as_crop'] += int(confusion_counts.get('weed_as_crop', 0))
            confusion_totals['crop_as_weed'] += int(confusion_counts.get('crop_as_weed', 0))
            removed_stats = summary.get('removed_pred_islands', {})
            removed_pred_islands_total['crop'] += int(removed_stats.get('crop', 0))
            removed_pred_islands_total['weed'] += int(removed_stats.get('weed', 0))

        metrics = OrderedDict()
        for class_name in classes:
            for size_name in sizes:
                metrics[f'inst_recall_{class_name}_{size_name}_total'] = float(
                    recall_totals[class_name][size_name]['total'])
                metrics[f'inst_precision_{class_name}_{size_name}_total_pred'] = float(
                    precision_totals[class_name][size_name]['total_pred'])
                for thr_pct in thrs_pct:
                    recall_val = self._safe_ratio(
                        recall_totals[class_name][size_name]['detected'][thr_pct],
                        recall_totals[class_name][size_name]['total']) * 100.0
                    tp_val = precision_totals[class_name][size_name]['tp'][thr_pct]
                    fp_val = precision_totals[class_name][size_name]['fp_pure'][thr_pct]
                    precision_denom = tp_val + fp_val
                    precision_val = self._safe_ratio(
                        tp_val,
                        precision_denom) * 100.0
                    metrics[f'inst_recall_{class_name}_{size_name}_{thr_pct}pct'] = recall_val
                    metrics[f'inst_precision_{class_name}_{size_name}_{thr_pct}pct'] = precision_val
                    metrics[f'inst_precision_{class_name}_{size_name}_{thr_pct}pct_tp'] = float(tp_val)
                    metrics[f'inst_precision_{class_name}_{size_name}_{thr_pct}pct_fp_pure'] = float(fp_val)

        metrics['inst_pred_removed_crop'] = float(removed_pred_islands_total['crop'])
        metrics['inst_pred_removed_weed'] = float(removed_pred_islands_total['weed'])
        metrics['inst_pred_removed_total'] = float(
            removed_pred_islands_total['crop'] + removed_pred_islands_total['weed'])
        metrics['inst_confusion_weed_as_crop'] = float(confusion_totals['weed_as_crop'])
        metrics['inst_confusion_crop_as_weed'] = float(confusion_totals['crop_as_weed'])
        metrics['inst_confusion_total'] = float(
            confusion_totals['weed_as_crop'] + confusion_totals['crop_as_weed'])

        self._log_summary_tables(
            logger,
            recall_totals=recall_totals,
            precision_totals=precision_totals,
            confusion_totals=confusion_totals,
            removed_pred_islands_total=removed_pred_islands_total)

        rounded_metrics = OrderedDict()
        for key, value in metrics.items():
            if key.endswith('pct'):
                rounded_metrics[key] = round(value, 2)
            else:
                rounded_metrics[key] = value

        self._log_visualization_audit(
            logger=logger,
            selected_total=vis_selected_total,
            selected_detected=vis_selected_detected)

        self._vis_initialized = False
        self._vis_sample_index = 0
        self._vis_small_pair_count = 0
        self._vis_count_by_key = OrderedDict()
        self._vis_manifest = []
        self._vis_fp_case_count = 0
        return rounded_metrics

    def _summarize_sample_records(self,
                                  instance_records: List[dict],
                                  pred_component_records: List[dict],
                                  confusion_counts: Optional[Dict[str, int]] = None) -> dict:
        classes = ('crop', 'weed')
        sizes = self._size_buckets
        thrs_pct = self.overlap_thrs_pct

        recall = {
            cls: {
                size: {'total': 0, 'detected': OrderedDict((t, 0) for t in thrs_pct)}
                for size in sizes
            }
            for cls in classes
        }
        precision = {
            cls: {
                size: {
                    'total_pred': 0,
                    'tp': OrderedDict((t, 0) for t in thrs_pct),
                    'fp_pure': OrderedDict((t, 0) for t in thrs_pct),
                }
                for size in sizes
            }
            for cls in classes
        }
        removed_pred_islands = self._pred_island_filter_stats(pred_component_records)

        vis_selected_total = 0
        vis_selected_detected = 0

        for record in instance_records:
            class_name = record['class_name']
            bin_name = record['bin_name']
            size_name = self._size_bucket(int(record['area']))
            overlap = float(record['overlap'])
            recall[class_name][size_name]['total'] += 1
            for thr_pct, thr in zip(thrs_pct, self._overlap_thrs):
                if overlap >= thr:
                    recall[class_name][size_name]['detected'][thr_pct] += 1

            if self.vis_class == 'all' or class_name == self.vis_class:
                if self.vis_area_bins is None or bin_name in self.vis_area_bins:
                    vis_selected_total += 1
                    vis_selected_detected += int(overlap >= self.primary_overlap_thr)

        gt_hits = {
            thr_pct: {
                'crop': {'le100': set(), 'ge100': set()},
                'weed': {'le100': set(), 'ge100': set()},
            }
            for thr_pct in thrs_pct
        }

        for record in pred_component_records:
            class_name = record['class_name']
            pred_size_name = self._size_bucket(int(record['area']))
            precision[class_name][pred_size_name]['total_pred'] += 1

            gt_overlap_items = record.get('gt_overlaps', ())
            for thr_pct, thr in zip(thrs_pct, self._overlap_thrs):
                hit_any_gt = False
                for gt_id, gt_area, gt_overlap in gt_overlap_items:
                    if float(gt_overlap) > thr:
                        hit_any_gt = True
                        gt_size_name = self._size_bucket(int(gt_area))
                        gt_hits[thr_pct][class_name][gt_size_name].add(int(gt_id))

                if not hit_any_gt:
                    precision[class_name][pred_size_name]['fp_pure'][thr_pct] += 1

        for class_name in classes:
            for size_name in sizes:
                for thr_pct in thrs_pct:
                    precision[class_name][size_name]['tp'][thr_pct] = len(
                        gt_hits[thr_pct][class_name][size_name])

        return dict(
            recall=recall,
            precision=precision,
            confusion_counts=dict(confusion_counts or {'weed_as_crop': 0, 'crop_as_weed': 0}),
            removed_pred_islands=removed_pred_islands,
            vis_selected_total=vis_selected_total,
            vis_selected_detected=vis_selected_detected,
        )

    def _compute_instance_class_confusions(self,
                                           pred_label: np.ndarray,
                                           instance_records: List[dict]) -> Tuple[List[dict], Dict[str, int]]:
        """Compute GT-instance class confusions using dominant predicted class.

        A GT instance is marked confused when the dominant predicted class among
        {crop, weed} pixels inside that GT mask is the opposite class.
        """
        confusion_records = []
        counts = {'weed_as_crop': 0, 'crop_as_weed': 0}

        for record in instance_records:
            inst_mask = record['inst_mask']
            area = int(record['area'])
            if area <= 0:
                continue

            gt_class_name = record['class_name']
            gt_class_label = int(record['class_label'])
            opp_label = self.weed_label if gt_class_label == self.crop_label else self.crop_label
            opp_name = self._label_to_name.get(opp_label, None)
            if opp_name is None:
                continue

            same_cnt = int(np.logical_and(inst_mask, pred_label == gt_class_label).sum())
            opp_cnt = int(np.logical_and(inst_mask, pred_label == opp_label).sum())
            if opp_cnt <= same_cnt or opp_cnt <= 0:
                continue

            ys, xs = np.where(inst_mask)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            opp_frac = opp_cnt / float(area)

            confusion_records.append(dict(
                instance_id=int(record['instance_id']),
                gt_class=gt_class_name,
                pred_class=opp_name,
                area=area,
                opp_frac=opp_frac,
                bbox=(x1, y1, x2, y2),
            ))

            if gt_class_name == 'weed' and opp_name == 'crop':
                counts['weed_as_crop'] += 1
            elif gt_class_name == 'crop' and opp_name == 'weed':
                counts['crop_as_weed'] += 1

        return confusion_records, counts

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
                    detected=overlap >= self.primary_overlap_thr,
                    overlap=overlap,
                    bin_name=bin_name,
                    inst_mask=inst_mask,
                ))

        return sample_stats

    def _evaluate_predicted_components(self, pred_label: np.ndarray, gt_label: np.ndarray,
                                       instance_map: np.ndarray) -> List[dict]:
        """Find predicted connected components and classify each as TP or FP.

        Matching is greedy and one-to-one: predicted components are sorted by
        area (largest first), and each GT instance can be claimed once.

        Weed components are additionally assigned to a matched GT size group:
        - ``lt100`` when matched GT area < 100 px and overlap >= threshold
        - ``ge100`` when matched GT area >= 100 px and overlap >= threshold

        Optional class-specific filtering can remove small predicted islands
        before matching.
        """
        pred_components = []
        removed_by_class = {'crop': 0, 'weed': 0}

        # Build GT instance lookup by semantic class for one-to-one matching.
        gt_by_class = {
            self.crop_label: {},
            self.weed_label: {},
        }
        instance_ids = np.unique(instance_map)
        instance_ids = instance_ids[instance_ids > 0]
        for instance_id in instance_ids:
            inst_mask = instance_map == instance_id
            inst_area = int(inst_mask.sum())
            if inst_area <= 0:
                continue
            class_name, class_label = self._instance_class(inst_mask, gt_label)
            if class_name is None:
                continue
            gt_by_class[class_label][int(instance_id)] = dict(mask=inst_mask, area=inst_area)

        for class_label, class_name in self._label_to_name.items():
            pred_cls_mask = pred_label == class_label
            if class_name == 'weed' and self.pred_weed_morph_kernel_size > 0:
                pred_cls_mask = self._apply_pred_weed_morphology(pred_cls_mask)
            if not pred_cls_mask.any():
                continue

            labeled_arr, n_components = sp_ndimage.label(pred_cls_mask)
            if n_components <= 0:
                continue

            # Greedy matching by predicted island size (largest first).
            comp_areas = np.bincount(labeled_arr.ravel())[1:]
            comp_ids = np.arange(1, n_components + 1)

            removed_for_class = 0
            min_area = int(max(0, self.pred_island_min_area_by_class.get(class_name, 0)))
            if self.pred_island_filter_enable and min_area > 0:
                keep_mask = comp_areas >= min_area
                removed_for_class = int(np.sum(~keep_mask))
                comp_ids = comp_ids[keep_mask]

            removed_by_class[class_name] = removed_for_class

            if comp_ids.size == 0:
                continue

            sorted_comp_ids = comp_ids[np.argsort(-comp_areas[comp_ids - 1])]
            claimed_gt_ids = set()

            for comp_id in sorted_comp_ids:
                comp_mask = labeled_arr == comp_id
                comp_area = int(comp_areas[comp_id - 1])
                if comp_area <= 0:
                    continue

                overlapping_ids = np.unique(instance_map[comp_mask])
                overlapping_ids = overlapping_ids[overlapping_ids > 0]

                best_overlap = 0.0
                best_gt_id = None
                best_gt_area = None
                gt_overlaps = []

                for inst_id in overlapping_ids:
                    inst_id = int(inst_id)
                    gt_info = gt_by_class[class_label].get(inst_id, None)
                    if gt_info is None:
                        continue

                    inst_mask = gt_info['mask']
                    inter = int(np.logical_and(comp_mask, inst_mask).sum())
                    if inter <= 0:
                        continue

                    if self.overlap_mode == 'gt':
                        # Use GT-normalized overlap for matching consistency with recall.
                        overlap = inter / float(gt_info['area'])
                    else:
                        union = int(np.logical_or(comp_mask, inst_mask).sum())
                        overlap = inter / float(union) if union > 0 else 0.0

                    overlap_gt = inter / float(gt_info['area'])
                    gt_overlaps.append((inst_id, int(gt_info['area']), overlap_gt))

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_gt_id = inst_id
                        best_gt_area = int(gt_info['area'])

                true_positive = (
                    best_gt_id is not None
                    and best_overlap >= self.primary_overlap_thr
                    and best_gt_id not in claimed_gt_ids)

                if true_positive:
                    claimed_gt_ids.add(best_gt_id)

                weed_match_group = None
                if (class_name == 'weed' and best_gt_id is not None
                    and best_overlap >= self.primary_overlap_thr):
                    weed_match_group = 'le100' if int(best_gt_area) <= 100 else 'gt100'

                bin_name = self._area_to_bin(comp_area)
                pred_components.append(dict(
                    class_name=class_name,
                    class_label=class_label,
                    area=comp_area,
                    true_positive=true_positive,
                    overlap=best_overlap,
                    bin_name=bin_name,
                    matched_instance_id=best_gt_id,
                    matched_gt_area=best_gt_area,
                    gt_overlaps=tuple(gt_overlaps),
                    weed_match_group=weed_match_group,
                    pred_island_removed_for_class=removed_for_class,
                ))

            self._last_pred_island_removed_by_class = removed_by_class
        return pred_components

    def _pred_island_filter_stats(self, pred_component_records: List[dict]) -> Dict[str, int]:
        removed = {'crop': 0, 'weed': 0}
        if not self.pred_island_filter_enable:
            return removed
        return dict(self._last_pred_island_removed_by_class)

    def _resolve_img_path(self, data_sample: dict) -> Optional[str]:
        img_path = data_sample.get('img_path', None)
        if img_path is None and hasattr(data_sample, 'metainfo'):
            img_path = data_sample.metainfo.get('img_path', None)
        return img_path

    def _load_rgb_image(self, img_path: str) -> np.ndarray:
        return np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

    def _apply_pred_weed_morphology(self, pred_mask: np.ndarray) -> np.ndarray:
        """Apply optional morphology to predicted weed mask before CCL."""
        k = self.pred_weed_morph_kernel_size
        if k <= 0:
            return np.asarray(pred_mask, dtype=bool)

        mask = np.asarray(pred_mask, dtype=bool)
        structure = np.ones((k, k), dtype=bool)
        if self.pred_weed_morph_op == 'dilation':
            return sp_ndimage.binary_dilation(mask, structure=structure)
        if self.pred_weed_morph_op == 'closing':
            return sp_ndimage.binary_closing(mask, structure=structure)
        return sp_ndimage.binary_erosion(mask, structure=structure)

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

            # Extra qualitative output for small weeds: GT (left) vs prediction (right).
            if (class_name == 'weed'
                    and int(record['area']) < 100
                    and self._vis_small_pair_count < self.vis_small_weed_pair_max):
                pair_dir = osp.join(vis_root, 'small_weed_pairs')
                mkdir_or_exist(pair_dir)
                pair_img = self._compose_small_weed_pair_visual(
                    rgb_image=rgb_image,
                    pred_label=pred_label,
                    record=record)
                pair_name = (
                    f'{self._vis_small_pair_count:04d}_{basename}_inst{record["instance_id"]}'
                    f'_area{record["area"]}.png')
                Image.fromarray(pair_img).save(osp.join(pair_dir, pair_name))
                self._vis_small_pair_count += 1

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

    def _compose_weed_island_precision_visual(self,
                                              rgb_image: np.ndarray,
                                              pred_label: np.ndarray,
                                              gt_label: np.ndarray) -> np.ndarray:
        """Full-image side-by-side weed island precision visualization.

        Left panel: GT on black background — crop pixels green, weed pixels red.
        Right panel: predicted weed islands on black background — area <= 100 px
            orange (with bright borders), area > 100 px blue. Crops shown as green.
        """
        h, w = gt_label.shape
        left = np.zeros((h, w, 3), dtype=np.uint8)  # Black canvas
        right = np.zeros((h, w, 3), dtype=np.uint8)  # Black canvas

        # --- Left: GT semantic overlay on black ---
        crop_mask = gt_label == self.crop_label
        weed_mask_gt = gt_label == self.weed_label
        crop_color = np.array([0, 255, 0], dtype=np.uint8)
        weed_gt_color = np.array([255, 0, 0], dtype=np.uint8)

        if crop_mask.any():
            left[crop_mask] = crop_color
        if weed_mask_gt.any():
            left[weed_mask_gt] = weed_gt_color 

        # --- Right: GT crops + predicted weed islands on black ---
        pred_crop_mask = pred_label == self.crop_label
        if pred_crop_mask.any():
            right[pred_crop_mask] = crop_color

        pred_weed_mask = pred_label == self.weed_label
        if self.pred_weed_morph_kernel_size > 0:
            pred_weed_mask = self._apply_pred_weed_morphology(pred_weed_mask)
        small_px_total = 0
        large_px_total = 0
        small_island_count = 0
        large_island_count = 0
        if pred_weed_mask.any():
            labeled_arr, n_components = sp_ndimage.label(pred_weed_mask)
            if n_components > 0:
                comp_areas = np.bincount(labeled_arr.ravel())[1:]
                small_color = np.array([255, 165, 0], dtype=np.uint8)  # orange: <= 100 px
                large_color = np.array([0, 0, 255], dtype=np.uint8)    # blue: > 100 px
                small_edge_color = np.array([255, 255, 0], dtype=np.uint8)  # bright yellow border
                for comp_id in range(1, n_components + 1):
                    area = int(comp_areas[comp_id - 1])
                    if area <= 0:
                        continue
                    comp_mask = labeled_arr == comp_id
                    is_small = area <= 100
                    color = small_color if is_small else large_color
                    right[comp_mask] = color

                    if is_small:
                        small_px_total += area
                        small_island_count += 1
                        # Highlight tiny islands with a bright contour for visibility.
                        edge_mask = np.logical_and(
                            comp_mask,
                            np.logical_not(sp_ndimage.binary_erosion(comp_mask)))
                        right[edge_mask] = small_edge_color
                    else:
                        large_px_total += area
                        large_island_count += 1

        pair = np.concatenate([left, right], axis=1)
        pil_img = Image.fromarray(pair)
        drawer = ImageDraw.Draw(pil_img)
        h, w = pair.shape[:2]
        mid_x = w // 2
        drawer.line([(mid_x, 0), (mid_x, h - 1)], fill=(255, 255, 255), width=2)
        # Left label
        drawer.rectangle([(0, 0), (260, 18)], fill=(0, 0, 0))
        drawer.text((4, 2), 'GT  (green=crop | red=weed)', fill=(255, 255, 255))
        # Right label
        drawer.rectangle([(mid_x, 0), (mid_x + 380, 18)], fill=(0, 0, 0))
        drawer.text((mid_x + 4, 2), 'Pred weed islands  (orange=<=100px | blue=>100px)',
                    fill=(255, 255, 255))
        drawer.rectangle([(mid_x, 18), (mid_x + 560, 38)], fill=(0, 0, 0))
        drawer.text(
            (mid_x + 4, 20),
            ('<=100px: '
             f'{small_px_total} px ({small_island_count} islands) | '
             '>100px: '
             f'{large_px_total} px ({large_island_count} islands)'),
            fill=(255, 255, 255))
        return np.array(pil_img, dtype=np.uint8)

    def _compose_small_weed_pair_visual(self,
                                        rgb_image: np.ndarray,
                                        pred_label: np.ndarray,
                                        record: dict) -> np.ndarray:
        left = np.asarray(rgb_image, dtype=np.uint8).copy()
        right = np.asarray(rgb_image, dtype=np.uint8).copy()

        inst_mask = record['inst_mask']
        pred_mask_cls = pred_label == record['class_label']

        gt_color = np.array([255, 0, 0], dtype=np.uint8)
        pred_color = np.array([0, 0, 255], dtype=np.uint8)

        left[inst_mask] = (
            (1.0 - self.vis_gt_alpha) * left[inst_mask]
            + self.vis_gt_alpha * gt_color
        ).astype(np.uint8)

        right[pred_mask_cls] = (
            (1.0 - self.vis_pred_alpha) * right[pred_mask_cls]
            + self.vis_pred_alpha * pred_color
        ).astype(np.uint8)

        pair = np.concatenate([left, right], axis=1)
        pil_img = Image.fromarray(pair)
        drawer = ImageDraw.Draw(pil_img)
        h, w = pair.shape[:2]
        mid_x = w // 2
        drawer.line([(mid_x, 0), (mid_x, h - 1)], fill=(255, 255, 255), width=2)
        drawer.rectangle([(0, 0), (150, 16)], fill=(0, 0, 0))
        drawer.text((4, 2), 'GT (<100 weed)', fill=(255, 255, 255))
        drawer.rectangle([(mid_x, 0), (mid_x + 180, 16)], fill=(0, 0, 0))
        drawer.text((mid_x + 4, 2), 'Prediction (weed)', fill=(255, 255, 255))
        return np.array(pil_img, dtype=np.uint8)

    def _analyze_weed_islands_for_fp_visual(self,
                                            pred_label: np.ndarray,
                                            gt_label: np.ndarray,
                                            overlap_thr: float) -> List[dict]:
        pred_weed_mask = pred_label == self.weed_label
        if self.pred_weed_morph_kernel_size > 0:
            pred_weed_mask = self._apply_pred_weed_morphology(pred_weed_mask)

        labeled_arr, n_components = sp_ndimage.label(pred_weed_mask)
        if n_components <= 0:
            return []

        gt_weed_mask = gt_label == self.weed_label
        gt_weed_labeled, gt_n = sp_ndimage.label(gt_weed_mask)
        if gt_n <= 0:
            gt_areas = np.array([], dtype=np.int64)
        else:
            gt_areas = np.bincount(gt_weed_labeled.ravel())[1:]

        components = []
        for comp_id in range(1, n_components + 1):
            comp_mask = labeled_arr == comp_id
            pred_area = int(comp_mask.sum())
            if pred_area <= 0:
                continue

            overlap_gt_ids = np.unique(gt_weed_labeled[comp_mask])
            overlap_gt_ids = overlap_gt_ids[overlap_gt_ids > 0]

            hit = False
            best_gt_overlap = 0.0
            inter_weed_px = int(np.logical_and(comp_mask, gt_weed_mask).sum())
            island_precision = inter_weed_px / float(pred_area)

            for gt_id in overlap_gt_ids:
                gt_id = int(gt_id)
                gt_area = int(gt_areas[gt_id - 1])
                if gt_area <= 0:
                    continue
                inter = int(np.logical_and(comp_mask, gt_weed_labeled == gt_id).sum())
                ov = inter / float(gt_area)
                if ov > overlap_thr:
                    hit = True
                if ov > best_gt_overlap:
                    best_gt_overlap = ov

            ys, xs = np.where(comp_mask)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            components.append(dict(
                mask=comp_mask,
                area=pred_area,
                bbox=(x1, y1, x2, y2),
                is_tp=bool(hit),
                island_precision=float(island_precision),
                best_gt_overlap=float(best_gt_overlap),
            ))

        return components

    def _compose_fp_case_visual(self,
                                rgb_image: np.ndarray,
                                pred_label: np.ndarray,
                                gt_label: np.ndarray,
                                overlap_thr: float,
                                confusion_records: Optional[List[dict]] = None) -> Optional[np.ndarray]:
        """Compose side-by-side FP-focused visualization for weed islands.

        Left: GT semantic overlay on alpha-dimmed background.
        Right: prediction semantic overlay and weed island TP/FP labels.
        Returns None when there is no FP island in the image.
        """
        islands = self._analyze_weed_islands_for_fp_visual(
            pred_label=pred_label,
            gt_label=gt_label,
            overlap_thr=overlap_thr)
        if len(islands) == 0:
            return None

        n_fp = sum(0 if item['is_tp'] else 1 for item in islands)
        if n_fp <= 0:
            return None

        base = np.asarray(rgb_image, dtype=np.uint8)
        dim_bg = np.clip(base.astype(np.float32) * self.vis_bg_alpha, 0, 255).astype(np.uint8)

        # Left panel: GT semantic mask with alpha on dimmed background.
        left = dim_bg.copy()
        crop_gt = gt_label == self.crop_label
        weed_gt = gt_label == self.weed_label
        crop_color = np.array([0, 255, 0], dtype=np.uint8)
        weed_color = np.array([255, 0, 0], dtype=np.uint8)
        if crop_gt.any():
            left[crop_gt] = (
                (1.0 - self.vis_gt_alpha) * left[crop_gt] + self.vis_gt_alpha * crop_color
            ).astype(np.uint8)
        if weed_gt.any():
            left[weed_gt] = (
                (1.0 - self.vis_gt_alpha) * left[weed_gt] + self.vis_gt_alpha * weed_color
            ).astype(np.uint8)

        # Right panel: prediction semantic map with island TP/FP coloring.
        right = dim_bg.copy()
        crop_pred = pred_label == self.crop_label
        weed_pred = pred_label == self.weed_label
        if crop_pred.any():
            right[crop_pred] = (
                (1.0 - self.vis_pred_alpha) * right[crop_pred] + self.vis_pred_alpha * crop_color
            ).astype(np.uint8)
        if weed_pred.any():
            right[weed_pred] = (
                (1.0 - self.vis_pred_alpha) * right[weed_pred] + self.vis_pred_alpha * np.array([255, 255, 255], dtype=np.uint8)
            ).astype(np.uint8)

        tp_color = np.array([0, 0, 255], dtype=np.uint8)       # blue
        fp_color = np.array([255, 165, 0], dtype=np.uint8)     # orange

        for comp in islands:
            color = tp_color if comp['is_tp'] else fp_color
            right[comp['mask']] = (
                (1.0 - self.vis_pred_alpha) * right[comp['mask']] + self.vis_pred_alpha * color
            ).astype(np.uint8)

        pair = np.concatenate([left, right], axis=1)
        pil_img = Image.fromarray(pair)
        drawer = ImageDraw.Draw(pil_img)
        h, w = pair.shape[:2]
        mid_x = w // 2
        drawer.line([(mid_x, 0), (mid_x, h - 1)], fill=(255, 255, 255), width=2)
        drawer.rectangle([(0, 0), (260, 18)], fill=(0, 0, 0))
        drawer.text((4, 2), 'GT (green=crop | red=weed)', fill=(255, 255, 255))
        drawer.rectangle([(mid_x, 0), (mid_x + 420, 18)], fill=(0, 0, 0))
        drawer.text((mid_x + 4, 2),
                    f'Pred (weed: blue=TP, orange=FP) @thr>{int(round(overlap_thr * 100.0))}%',
                    fill=(255, 255, 255))

        # Draw island labels on right panel.
        for idx, comp in enumerate(islands, start=1):
            x1, y1, x2, y2 = comp['bbox']
            rx1 = x1 + mid_x
            rx2 = x2 + mid_x
            status = 'TP' if comp['is_tp'] else 'FP'
            edge_color = (0, 128, 255) if comp['is_tp'] else (255, 140, 0)
            drawer.rectangle([(rx1, y1), (rx2, y2)], outline=edge_color, width=2)

            text = (
                f'#{idx} {status} | P={comp["island_precision"] * 100.0:.1f}% '
                f'| ov={comp["best_gt_overlap"] * 100.0:.1f}%')
            ty = max(20, y1 - 14)
            tw = min(rx1 + 300, w - 1)
            drawer.rectangle([(rx1, ty), (tw, ty + 14)], fill=(0, 0, 0))
            drawer.text((rx1 + 2, ty), text, fill=(255, 255, 255))

        # Mark GT-instance class confusions explicitly.
        confusion_records = confusion_records or []
        n_weed_as_crop = 0
        n_crop_as_weed = 0
        for conf in confusion_records:
            x1, y1, x2, y2 = conf['bbox']
            gt_cls = conf['gt_class']
            pred_cls = conf['pred_class']
            frac = float(conf['opp_frac']) * 100.0
            label = f'{gt_cls}->{pred_cls} ({frac:.1f}%)'
            if gt_cls == 'weed' and pred_cls == 'crop':
                n_weed_as_crop += 1
                edge_left = (255, 165, 0)
                edge_right = (255, 165, 0)
                semantics = 'FN weed / FP crop'
            else:
                n_crop_as_weed += 1
                edge_left = (0, 255, 255)
                edge_right = (0, 255, 255)
                semantics = 'FN crop / FP weed'

            # Left panel: highlight GT confused instance.
            drawer.rectangle([(x1, y1), (x2, y2)], outline=edge_left, width=2)
            ly = max(20, y1 - 14)
            ltw = min(x1 + 320, mid_x - 1)
            drawer.rectangle([(x1, ly), (ltw, ly + 14)], fill=(0, 0, 0))
            drawer.text((x1 + 2, ly), f'{label} | {semantics}', fill=(255, 255, 255))

            # Right panel: project same GT box for quick reference.
            rx1 = x1 + mid_x
            rx2 = x2 + mid_x
            drawer.rectangle([(rx1, y1), (rx2, y2)], outline=edge_right, width=2)

        footer = (
            f'Islands: {len(islands)} | FP: {n_fp} | '
            f'weed->crop: {n_weed_as_crop} | crop->weed: {n_crop_as_weed}')
        drawer.rectangle([(mid_x, 18), (min(mid_x + 520, w - 1), 36)], fill=(0, 0, 0))
        drawer.text((mid_x + 4, 20), footer, fill=(255, 255, 255))

        return np.array(pil_img, dtype=np.uint8)

    def _log_visualization_audit(self,
                                 logger: MMLogger,
                                 selected_total: Optional[int] = None,
                                 selected_detected: Optional[int] = None,
                                 results: Optional[list] = None) -> None:
        if self.vis_output_dir is None:
            return

        vis_root = self._vis_eval_output_dir or self.vis_output_dir

        if len(self._vis_manifest) > 0:
            manifest_path = osp.join(vis_root, 'manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump(self._vis_manifest, f, indent=2)
            print_log(f'instance detection visualization manifest: {manifest_path}', logger=logger)

        if selected_total is None or selected_detected is None:
            selected_total = 0
            selected_detected = 0
            for item in (results or []):
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

    @staticmethod
    def _size_bucket(area: int) -> str:
        return 'le100' if area <= 100 else 'ge100'

    @staticmethod
    def _size_bucket_label(size_bucket: str) -> str:
        if size_bucket == 'le100':
            return '<=100'
        return '>=100'

    def _log_summary_tables(self,
                            logger: MMLogger,
                            recall_totals: Dict[str, Dict[str, Dict[str, OrderedDict]]],
                            precision_totals: Dict[str, Dict[str, Dict[str, OrderedDict]]],
                            confusion_totals: Optional[Dict[str, int]] = None,
                            removed_pred_islands_total: Optional[Dict[str, int]] = None) -> None:
        thr_cols = [f'{thr_pct}%' for thr_pct in self.overlap_thrs_pct]

        recall_table = PrettyTable()
        recall_table.field_names = ['Class', 'Size (px)', 'Total'] + [f'Recall@{c}' for c in thr_cols]
        for class_name in ('crop', 'weed'):
            for size_bucket in self._size_buckets:
                row = [
                    class_name,
                    self._size_bucket_label(size_bucket),
                    recall_totals[class_name][size_bucket]['total'],
                ]
                for thr_pct in self.overlap_thrs_pct:
                    det = recall_totals[class_name][size_bucket]['detected'][thr_pct]
                    tot = recall_totals[class_name][size_bucket]['total']
                    row.append(np.round(self._safe_ratio(det, tot) * 100.0, 2))
                recall_table.add_row(row)

        precision_table = PrettyTable()
        precision_table.field_names = [
            'Class', 'Size (px)', 'Total Pred',
        ] + [f'TP@{c}' for c in thr_cols] + [f'FP_pure@{c}' for c in thr_cols] + [
            f'Precision@{c}' for c in thr_cols
        ]
        for class_name in ('crop', 'weed'):
            for size_bucket in self._size_buckets:
                row = [
                    class_name,
                    self._size_bucket_label(size_bucket),
                    precision_totals[class_name][size_bucket]['total_pred'],
                ]
                for thr_pct in self.overlap_thrs_pct:
                    tp = precision_totals[class_name][size_bucket]['tp'][thr_pct]
                    row.append(tp)
                for thr_pct in self.overlap_thrs_pct:
                    fp_pure = precision_totals[class_name][size_bucket]['fp_pure'][thr_pct]
                    row.append(fp_pure)
                for thr_pct in self.overlap_thrs_pct:
                    tp = precision_totals[class_name][size_bucket]['tp'][thr_pct]
                    fp_pure = precision_totals[class_name][size_bucket]['fp_pure'][thr_pct]
                    row.append(np.round(self._safe_ratio(tp, tp + fp_pure) * 100.0, 2))
                precision_table.add_row(row)

        print_log('instance detection summary (recall by size, thresholds in %):', logger=logger)
        print_log('\n' + recall_table.get_string(), logger=logger)
        print_log('instance detection summary (precision by size, thresholds in %):', logger=logger)
        print_log('\n' + precision_table.get_string(), logger=logger)

        confusion_totals = confusion_totals or {'weed_as_crop': 0, 'crop_as_weed': 0}
        confusion_table = PrettyTable()
        confusion_table.field_names = [
            'GT Class', 'Predicted As', 'Count', 'Interpretation'
        ]
        confusion_table.add_row([
            'weed',
            'crop',
            int(confusion_totals.get('weed_as_crop', 0)),
            'FN for weed, FP for crop',
        ])
        confusion_table.add_row([
            'crop',
            'weed',
            int(confusion_totals.get('crop_as_weed', 0)),
            'FN for crop, FP for weed',
        ])
        confusion_table.add_row([
            'total',
            'cross-class',
            int(confusion_totals.get('weed_as_crop', 0)) + int(confusion_totals.get('crop_as_weed', 0)),
            'counts GT instances with dominant opposite label',
        ])
        print_log('instance detection class-confusion summary:', logger=logger)
        print_log('\n' + confusion_table.get_string(), logger=logger)

        if self.pred_island_filter_enable:
            removed_pred_islands_total = removed_pred_islands_total or {'crop': 0, 'weed': 0}
            print_log(
                'pred island filter stats: '
                f'removed_crop={int(removed_pred_islands_total.get("crop", 0))}, '
                f'removed_weed={int(removed_pred_islands_total.get("weed", 0))}, '
                f'removed_total={int(removed_pred_islands_total.get("crop", 0)) + int(removed_pred_islands_total.get("weed", 0))}',
                logger=logger)