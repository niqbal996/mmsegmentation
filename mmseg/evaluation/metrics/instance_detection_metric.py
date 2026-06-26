# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image, ImageDraw
from prettytable import PrettyTable
from scipy import ndimage as sp_ndimage

from mmseg.registry import METRICS


@METRICS.register_module()
class InstanceDetectionMetric(BaseMetric):
    """Instance-driven crop/weed metric with object and pixel sub-metrics.

    Sub-metrics:
                1) Object-level metrics per class (crop/weed):
           - Predicted weed connected-components are extracted from prediction.
           - If multiple predicted blobs overlap the same GT weed instance,
             they are merged as one virtual prediction for that GT instance.
                     - Object recall uses IoG:
                         IoG = Area(virtual_pred ∩ GT_instance) / Area(GT_instance).
                     - GT instance is recalled if IoG >= ``object_iog_thr``.
                     - Object precision uses IoP on each predicted component:
                         IoP = Area(pred_component ∩ GT_class_pixels) /
                                     Area(pred_component).
                     - Predicted component is precision-TP if IoP >= ``object_iop_thr``.
                     - Predicted component is precision-FP otherwise.

          2) Pixel-level mIoU over a fixed 3x3 confusion matrix:
              classes are [class0, class1, class2] by configured label ids.
           Per-class IoU/Precision/Recall/F1 are reported, along with means.

    Args:
        object_iog_thr (float, optional): IoG threshold for object recall.
            If None, falls back to ``object_ioa_thr`` for backward
            compatibility, then 0.10.
        object_iop_thr (float, optional): IoP threshold for object precision.
            If None, uses ``object_iog_thr``.
        object_ioa_thr (float, optional): Deprecated alias for
            ``object_iog_thr``.
        class0_label (int): Label id for class slot 0. Default: 0.
        class1_label (int): Label id for class slot 1. Default: 1.
        class2_label (int): Label id for class slot 2. Default: 2.
        class_names (Sequence[str]): Names for [class0, class1, class2].
            Example: ('background/soil', 'crop', 'weed').
        ignore_index (int): Ignore index in GT semantic mask. Default: 255.
        ignore_label_ids (Sequence[int], optional): Additional GT labels to
            ignore for pixel confusion and overlap checks.
        instance_map_path (str, optional): Optional root directory for
            instance maps when sample metainfo does not include a direct path.
        instance_map_suffix (str): Suffix for inferred instance map filename.
            Default: '.npz'.
        instance_map_subdirs (Sequence[str]): Candidate subdir names when
            inferring from semantic map path.
        allow_semantic_instance_fallback (bool): If instance map is missing,
            create pseudo instance map from GT weed connected-components.
            Default: True.
        resize_instance_to_gt (bool): Resize instance map to GT shape with
            nearest-neighbor when shape mismatch occurs. Default: True.
        vis_output_dir (str, optional): Output directory for FP-focused
            visualization images. If None, visualization is disabled.
        vis_fp_case_max (int): Maximum number of FP-focused visualization
            images to save. Default: 10.
        vis_bg_alpha (float): Background alpha for visualization panels.
            Default: 0.35.
        pred_island_min_area (int): Minimum predicted connected-component area
            (in pixels) kept for object-level TP/FP accounting. Components
            smaller than this threshold are ignored entirely. Set to 0 to
            disable filtering. Default: 0.
        exg_unlabelled_thr (float, optional): Excess Green index threshold
            (ExG = 2G − R − B) used to flag FP-on-background components as
            "probably unlabelled plants". For each predicted component that is
            FP on class0 (background/soil), the mean ExG over its pixels is
            computed from the RGB image. Components exceeding this threshold
            are counted in a separate "Prob.Unlabelled" column in the summary
            table. Does not affect any metric values. Set to None to disable.
            Default: 20.0.
        audit_thresholds (Sequence[float], optional): When set, a threshold
            sweep is performed at the end of ``compute_metrics`` without
            re-running inference. Each value is applied as both the IoG
            (recall) and IoP (precision) threshold simultaneously. Results
            are printed as a single table and, if ``audit_figure_path`` is
            given, as a matplotlib figure. Raw IoG/IoP scores are stored
            during ``process`` so no extra forward passes are needed.
            Example: ``[0.01, 0.05, 0.10, 0.20, 0.50, 1.0]``.
            Default: None (disabled).
        audit_figure_path (str, optional): File path (e.g. ``audit.png``)
            where the threshold-sweep precision/recall figure is saved.
            Requires ``audit_thresholds`` to be set. Default: None.
        collect_device (str): 'cpu' or 'gpu'. Default: 'cpu'.
        prefix (str, optional): Metric name prefix.
    """

    def __init__(self,
                 object_iog_thr: Optional[float] = None,
                 object_iop_thr: Optional[float] = None,
                 object_ioa_thr: Optional[float] = None,
                 class0_label: int = 0,
                 class1_label: int = 1,
                 class2_label: int = 2,
                 class_names: Sequence[str] = ('background/soil', 'crop', 'weed'),
                 ignore_index: int = 255,
                 ignore_label_ids: Optional[Sequence[int]] = None,
                 instance_map_path: Optional[str] = None,
                 instance_map_suffix: str = '.npz',
                 instance_map_subdirs: Sequence[str] = ('plant_instances',
                                                       'instance_segmentation'),
                 allow_semantic_instance_fallback: bool = True,
                 resize_instance_to_gt: bool = True,
                 vis_output_dir: Optional[str] = None,
                 vis_fp_case_max: int = 10,
                 vis_bg_alpha: float = 0.35,
                 pred_island_min_area: int = 0,
                 exg_unlabelled_thr: Optional[float] = 20.0,
                 audit_thresholds: Optional[Sequence[float]] = None,
                 audit_figure_path: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if object_iog_thr is None:
            if object_ioa_thr is not None:
                object_iog_thr = object_ioa_thr
            else:
                object_iog_thr = 0.10
        if object_iop_thr is None:
            object_iop_thr = object_iog_thr

        self.object_iog_thr = float(object_iog_thr)
        self.object_iop_thr = float(object_iop_thr)
        # Keep deprecated alias for backward compatibility in internal code.
        self.object_ioa_thr = self.object_iog_thr

        if not (0.0 <= self.object_iog_thr <= 1.0):
            raise ValueError(
                f'object_iog_thr must be in [0, 1], got {object_iog_thr}.')
        if not (0.0 <= self.object_iop_thr <= 1.0):
            raise ValueError(
                f'object_iop_thr must be in [0, 1], got {object_iop_thr}.')

        # Backward-compatible aliases from older config names.
        if 'background_label' in kwargs:
            class0_label = kwargs.pop('background_label')
        if 'soil_label' in kwargs:
            class1_label = kwargs.pop('soil_label')
        if 'crop_label' in kwargs:
            class1_label = kwargs.pop('crop_label')
        if 'weed_label' in kwargs:
            class2_label = kwargs.pop('weed_label')

        self.class0_label = int(class0_label)
        self.class1_label = int(class1_label)
        self.class2_label = int(class2_label)
        self.weed_label = self.class2_label

        if len(class_names) != 3:
            raise ValueError('class_names must contain exactly 3 entries.')
        self.class_names = tuple(str(x) for x in class_names)
        self.eval_label_ids = (
            self.class0_label,
            self.class1_label,
            self.class2_label,
        )
        if len(set(self.eval_label_ids)) != 3:
            raise ValueError(
                'class0_label, class1_label and class2_label must be unique '
                f'for a valid 3x3 confusion matrix, got {self.eval_label_ids}.')

        self.ignore_index = int(ignore_index)
        self.ignore_label_ids = set(int(x) for x in (ignore_label_ids or []))

        # Object-level evaluation is reported for crop (class1) and weed (class2).
        self.object_eval_class_labels = (self.class1_label, self.class2_label)

        self.instance_map_path = instance_map_path
        self.instance_map_suffix = str(instance_map_suffix)
        self.instance_map_subdirs = tuple(instance_map_subdirs)
        self.allow_semantic_instance_fallback = bool(allow_semantic_instance_fallback)
        self.resize_instance_to_gt = bool(resize_instance_to_gt)

        self.vis_output_dir = vis_output_dir
        self.vis_fp_case_max = int(max(0, vis_fp_case_max))
        self.vis_bg_alpha = float(max(0.0, min(1.0, vis_bg_alpha)))
        self.pred_island_min_area = int(max(0, pred_island_min_area))
        self.exg_unlabelled_thr = (float(exg_unlabelled_thr)
                                   if exg_unlabelled_thr is not None else None)
        self._vis_fp_candidates: List[dict] = []
        self._vis_case_seq = 0
        self.audit_thresholds = (
            sorted(float(t) for t in audit_thresholds)
            if audit_thresholds is not None else None
        )
        self.audit_figure_path = audit_figure_path
        if self.vis_output_dir is not None:
            mkdir_or_exist(self.vis_output_dir)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch and append compact per-sample summaries."""
        logger: MMLogger = MMLogger.get_current_instance()

        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            gt_label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)

            if pred_label.shape != gt_label.shape:
                print_log(
                    'InstanceDetectionMetric: shape mismatch detected, resizing '
                    f'prediction from {tuple(pred_label.shape)} to '
                    f'{tuple(gt_label.shape)} with nearest interpolation.',
                    logger=logger)
                pred_label = F.interpolate(
                    pred_label[None, None].float(),
                    size=gt_label.shape,
                    mode='nearest')[0, 0].to(gt_label.dtype)

            pred_np = pred_label.cpu().numpy().astype(np.int64)
            gt_np = gt_label.cpu().numpy().astype(np.int64)

            instance_map_file = self._resolve_instance_map_path(data_sample)
            try:
                instance_map = self._load_instance_map(instance_map_file)
            except FileNotFoundError:
                if not self.allow_semantic_instance_fallback:
                    raise
                print_log(
                    'InstanceDetectionMetric: instance map missing, falling '
                    'back to connected components from GT weed semantic mask: '
                    f'{instance_map_file}',
                    logger=logger)
                instance_map = self._semantic_to_instance_map(gt_np)

            if instance_map.shape != gt_np.shape:
                if not self.resize_instance_to_gt:
                    raise ValueError(
                        f'Shape mismatch for {instance_map_file}: '
                        f'instance_map={instance_map.shape}, gt={gt_np.shape}')
                instance_map = self._resize_instance_map_nearest(
                    instance_map, target_shape=gt_np.shape)

            rgb_np = None
            if self.exg_unlabelled_thr is not None:
                img_path = self._resolve_img_path(data_sample)
                if img_path and osp.exists(img_path):
                    rgb_np = self._load_rgb_image(img_path)
                    if rgb_np.shape[:2] != gt_np.shape:
                        rgb_np = self._resize_rgb_nearest(rgb_np, gt_np.shape)

            object_stats = self._compute_object_level_stats(
                pred_label=pred_np,
                gt_label=gt_np,
                instance_map=instance_map,
                rgb=rgb_np)
            pixel_confusion = self._compute_pixel_confusion_matrix(
                pred_label=pred_np,
                gt_label=gt_np)

            if self.vis_output_dir is not None and self.vis_fp_case_max > 0:
                self._collect_vis_fp_candidate(
                    data_sample=data_sample,
                    pred_label=pred_np,
                    gt_label=gt_np,
                    instance_map=instance_map)

            self.results.append(
                dict(
                    object_stats=object_stats,
                    pixel_confusion=pixel_confusion,
                    date_bin=self._extract_date_bin(
                        self._resolve_img_path(data_sample)),
                ))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Aggregate sample summaries and return final metrics."""
        logger: MMLogger = MMLogger.get_current_instance()

        if len(results) == 0:
            logger.warning('No samples available for InstanceDetectionMetric.')
            return OrderedDict()

        object_totals = self._init_object_totals()
        removed_totals = {
            class_label: {'all': 0.0, 'le100': 0.0, 'gt100': 0.0}
            for class_label in self.object_eval_class_labels
        }
        fp_breakdown_totals = self._init_fp_breakdown_totals()

        confusion = np.zeros((3, 3), dtype=np.int64)

        for item in results:
            self._accumulate_object_totals(object_totals, item['object_stats'])
            self._accumulate_removed_totals(removed_totals, item['object_stats'])
            self._accumulate_fp_breakdown_totals(
                fp_breakdown_totals, item['object_stats'])
            confusion += item['pixel_confusion']

        self._finalize_object_totals(object_totals)

        per_class = self._derive_per_class_from_confusion(confusion)
        mean_iou = float(np.nanmean([x['iou'] for x in per_class]))
        mean_precision = float(np.nanmean([x['precision'] for x in per_class]))
        mean_recall = float(np.nanmean([x['recall'] for x in per_class]))
        mean_f1 = float(np.nanmean([x['f1'] for x in per_class]))

        weed_idx = 2
        weed_iou = per_class[weed_idx]['iou']
        weed_precision = per_class[weed_idx]['precision']
        weed_recall = per_class[weed_idx]['recall']
        weed_f1 = per_class[weed_idx]['f1']

        weed_obj = object_totals[self.class2_label]['all']

        metrics = OrderedDict()
        metrics['n_images'] = len(results)

        # Backward-compatible aggregate object metrics (weed, all sizes).
        # tp/fp map to IoP precision counts; fn maps to IoG recall misses.
        metrics['obj_tp'] = float(weed_obj['tp_precision'])
        metrics['obj_fp'] = float(weed_obj['fp_precision'])
        metrics['obj_fn'] = float(weed_obj['fn_recall'])
        metrics['obj_gt_total'] = float(weed_obj['gt_total'])
        metrics['obj_pred_total'] = float(weed_obj['pred_total'])
        metrics['obj_precision'] = round(weed_obj['precision'] * 100.0, 2)
        metrics['obj_recall'] = round(weed_obj['recall'] * 100.0, 2)
        metrics['obj_iop_precision'] = round(weed_obj['precision'] * 100.0, 2)
        metrics['obj_iog_recall'] = round(weed_obj['recall'] * 100.0, 2)
        metrics['obj_f1'] = round(weed_obj['f1'] * 100.0, 2)

        metrics['pixel_mIoU'] = round(mean_iou * 100.0, 2)
        metrics['pixel_mPrecision'] = round(mean_precision * 100.0, 2)
        metrics['pixel_mRecall'] = round(mean_recall * 100.0, 2)
        metrics['pixel_mF1'] = round(mean_f1 * 100.0, 2)

        for idx, cls_name in enumerate(self.class_names):
            safe = cls_name.replace(' ', '_')
            metrics[f'pixel_IoU_{safe}'] = round(per_class[idx]['iou'] * 100.0, 2)
            metrics[f'pixel_Precision_{safe}'] = round(
                per_class[idx]['precision'] * 100.0, 2)
            metrics[f'pixel_Recall_{safe}'] = round(
                per_class[idx]['recall'] * 100.0, 2)
            metrics[f'pixel_F1_{safe}'] = round(per_class[idx]['f1'] * 100.0, 2)

        metrics['weed_iou'] = round(weed_iou * 100.0, 2)
        metrics['weed_precision'] = round(weed_precision * 100.0, 2)
        metrics['weed_recall'] = round(weed_recall * 100.0, 2)
        metrics['weed_f1'] = round(weed_f1 * 100.0, 2)

        weed_le100_bucket = object_totals[self.class2_label]['le100']
        _raw_prec_le100 = self._safe_div(
            weed_le100_bucket['pixel_inter_sum'],
            weed_le100_bucket['pixel_area_sum'])
        metrics['weed_pixel_prec_le100'] = round(
            float('nan') if np.isnan(_raw_prec_le100) else _raw_prec_le100 * 100.0, 2)

        for class_label in self.object_eval_class_labels:
            class_name = self._class_name_for_label(class_label)
            safe_cls = class_name.replace(' ', '_').replace('/', '_')
            for bin_key in ('all', 'le100', 'gt100'):
                obj_bin = object_totals[class_label][bin_key]
                key_prefix = f'obj_{safe_cls}_{bin_key}'
                metrics[f'{key_prefix}_gt_total'] = float(obj_bin['gt_total'])
                metrics[f'{key_prefix}_pred_total'] = float(obj_bin['pred_total'])
                metrics[f'{key_prefix}_tp'] = float(obj_bin['tp_precision'])
                metrics[f'{key_prefix}_fp'] = float(obj_bin['fp_precision'])
                metrics[f'{key_prefix}_fn'] = float(obj_bin['fn_recall'])
                metrics[f'{key_prefix}_tp_precision'] = float(obj_bin['tp_precision'])
                metrics[f'{key_prefix}_fp_precision'] = float(obj_bin['fp_precision'])
                metrics[f'{key_prefix}_tp_recall'] = float(obj_bin['tp_recall'])
                metrics[f'{key_prefix}_fn_recall'] = float(obj_bin['fn_recall'])
                metrics[f'{key_prefix}_precision'] = round(
                    obj_bin['precision'] * 100.0, 2)
                metrics[f'{key_prefix}_recall'] = round(
                    obj_bin['recall'] * 100.0, 2)
                metrics[f'{key_prefix}_iop_precision'] = round(
                    obj_bin['precision'] * 100.0, 2)
                metrics[f'{key_prefix}_iog_recall'] = round(
                    obj_bin['recall'] * 100.0, 2)
                metrics[f'{key_prefix}_f1'] = round(obj_bin['f1'] * 100.0, 2)
                metrics[f'{key_prefix}_removed_small'] = float(
                    removed_totals[class_label][bin_key])

                fp_bin = fp_breakdown_totals[class_label][bin_key]
                metrics[f'{key_prefix}_fp_edge_truncated'] = float(
                    fp_bin['edge_fp'])
                for gt_cat_key, gt_count in fp_bin['dominant_counts'].items():
                    metrics[f'{key_prefix}_fp_on_{gt_cat_key}'] = float(gt_count)
                metrics[f'{key_prefix}_fp_probably_unlabelled'] = float(
                    fp_bin.get('probably_unlabelled', 0.0))

        self._log_object_summary(
            logger=logger,
            object_totals=object_totals,
            pixel_metrics=per_class,
            fp_breakdown_totals=fp_breakdown_totals)
        self._log_paper_summary(
            logger=logger,
            metrics=metrics,
            n_images=len(results))
        self._log_date_breakdown_table(results, logger)
        self._dump_top_fp_visualizations(logger=logger)
        self._vis_fp_candidates = []
        self._vis_case_seq = 0

        if self.audit_thresholds is not None:
            self._run_threshold_audit(
                results, logger, per_class, fp_breakdown_totals)

        return metrics

    def _init_object_totals(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        out = {}
        for class_label in self.object_eval_class_labels:
            out[class_label] = {
                'all': self._new_count_bucket(),
                'le100': self._new_count_bucket(),
                'gt100': self._new_count_bucket(),
            }
        return out

    @staticmethod
    def _new_count_bucket() -> Dict[str, float]:
        return dict(
            gt_total=0.0,
            pred_total=0.0,
            tp_precision=0.0,
            fp_precision=0.0,
            tp_recall=0.0,
            fn_recall=0.0,
            precision=float('nan'),
            recall=float('nan'),
            f1=float('nan'),
            pixel_inter_sum=0.0,
            pixel_area_sum=0.0,
        )

    def _accumulate_object_totals(self,
                                  totals: Dict[int, Dict[str, Dict[str, float]]],
                                  sample_object: Dict[str, dict]) -> None:
        sample_per_class = sample_object.get('per_class', {})
        for class_label in self.object_eval_class_labels:
            sample_stats = sample_per_class.get(class_label, {})
            for bin_key in ('all', 'le100', 'gt100'):
                src = sample_stats.get(bin_key, None)
                if src is None:
                    continue
                dst = totals[class_label][bin_key]
                dst['gt_total'] += float(src.get('gt_total', 0.0))
                dst['pred_total'] += float(src.get('pred_total', 0.0))
                dst['tp_precision'] += float(src.get('tp_precision', 0.0))
                dst['fp_precision'] += float(src.get('fp_precision', 0.0))
                dst['tp_recall'] += float(src.get('tp_recall', 0.0))
                dst['fn_recall'] += float(src.get('fn_recall', 0.0))
                dst['pixel_inter_sum'] += float(src.get('pixel_inter_sum', 0.0))
                dst['pixel_area_sum'] += float(src.get('pixel_area_sum', 0.0))

    def _accumulate_removed_totals(self,
                                   removed_totals: Dict[int, Dict[str, float]],
                                   sample_object: Dict[str, dict]) -> None:
        sample_removed = sample_object.get('removed_small', {})
        for class_label in self.object_eval_class_labels:
            class_removed = sample_removed.get(class_label, {})
            for bin_key in ('all', 'le100', 'gt100'):
                removed_totals[class_label][bin_key] += float(
                    class_removed.get(bin_key, 0.0))

    def _init_fp_breakdown_totals(self) -> Dict[int, Dict[str, dict]]:
        out: Dict[int, Dict[str, dict]] = {}
        for class_label in self.object_eval_class_labels:
            out[class_label] = {}
            for bin_key in ('all', 'le100', 'gt100'):
                out[class_label][bin_key] = dict(
                    edge_fp=0.0,
                    dominant_counts=self._new_fp_dominant_counts(),
                    probably_unlabelled=0.0)
        return out

    def _new_fp_dominant_counts(self) -> Dict[str, float]:
        return {
            self._fp_cat_name(self.class0_label): 0.0,
            self._fp_cat_name(self.class1_label): 0.0,
            self._fp_cat_name(self.class2_label): 0.0,
            'ignore': 0.0,
            'other': 0.0,
        }

    def _accumulate_fp_breakdown_totals(self,
                                        totals: Dict[int, Dict[str, dict]],
                                        sample_object: Dict[str, dict]) -> None:
        sample_fp_meta = sample_object.get('fp_meta', {})
        for class_label in self.object_eval_class_labels:
            class_meta = sample_fp_meta.get(class_label, {})
            for bin_key in ('all', 'le100', 'gt100'):
                src = class_meta.get(bin_key, None)
                if src is None:
                    continue
                dst = totals[class_label][bin_key]
                dst['edge_fp'] += float(src.get('edge_fp', 0.0))
                src_dom = src.get('dominant_counts', {})
                dst_dom = dst['dominant_counts']
                for cat_key in dst_dom.keys():
                    dst_dom[cat_key] += float(src_dom.get(cat_key, 0.0))
                dst['probably_unlabelled'] += float(
                    src.get('probably_unlabelled', 0.0))

    def _finalize_object_totals(self,
                                totals: Dict[int, Dict[str, Dict[str, float]]]) -> None:
        for class_label in self.object_eval_class_labels:
            for bin_key in ('all', 'le100', 'gt100'):
                bucket = totals[class_label][bin_key]
                precision = self._safe_div(
                    bucket['tp_precision'], bucket['tp_precision'] + bucket['fp_precision'])
                recall = self._safe_div(
                    bucket['tp_recall'], bucket['tp_recall'] + bucket['fn_recall'])
                bucket['precision'] = precision
                bucket['recall'] = recall
                bucket['f1'] = self._f1(precision, recall)

    def _class_name_for_label(self, class_label: int) -> str:
        if class_label == self.class0_label:
            return self.class_names[0]
        if class_label == self.class1_label:
            return self.class_names[1]
        if class_label == self.class2_label:
            return self.class_names[2]
        return str(class_label)

    @staticmethod
    def _size_bin(area: int) -> str:
        # Keep legacy bin keys for compatibility, but use tiny < 100 pixels.
        return 'le100' if int(area) < 100 else 'gt100'

    def _fp_cat_name(self, class_label: int) -> str:
        return self._class_name_for_label(class_label).replace(' ', '_').replace('/', '_')

    def _dominant_gt_category(self,
                              comp_mask: np.ndarray,
                              gt_label: np.ndarray) -> str:
        gt_vals = gt_label[comp_mask]
        if gt_vals.size == 0:
            return 'other'
        uniq_vals, uniq_cnts = np.unique(gt_vals, return_counts=True)
        dom = int(uniq_vals[np.argmax(uniq_cnts)])
        if dom == self.class0_label:
            return self._fp_cat_name(self.class0_label)
        if dom == self.class1_label:
            return self._fp_cat_name(self.class1_label)
        if dom == self.class2_label:
            return self._fp_cat_name(self.class2_label)
        if dom == self.ignore_index or dom in self.ignore_label_ids:
            return 'ignore'
        return 'other'

    @staticmethod
    def _touches_image_edge(comp_mask: np.ndarray) -> bool:
        return bool(
            comp_mask[0, :].any() or comp_mask[-1, :].any() or
            comp_mask[:, 0].any() or comp_mask[:, -1].any())

    def _collect_vis_fp_candidate(self,
                                  data_sample: dict,
                                  pred_label: np.ndarray,
                                  gt_label: np.ndarray,
                                  instance_map: np.ndarray) -> None:
        img_path = self._resolve_img_path(data_sample)
        if img_path is None or (not osp.exists(img_path)):
            return

        analysis = self._analyze_weed_islands_for_visualization(
            pred_label=pred_label,
            gt_label=gt_label,
            instance_map=instance_map)
        small_fp_count = int(analysis['small_fp_count'])
        if small_fp_count <= 0:
            return

        candidate = dict(
            rank_score=small_fp_count,
            seq=self._vis_case_seq,
            img_path=img_path,
            pred_label=np.asarray(pred_label, dtype=np.int64),
            gt_label=np.asarray(gt_label, dtype=np.int64),
            comp_records=analysis['components'],
            fn_masks=analysis['fn_masks'],
            small_fp_count=small_fp_count,
            small_tp_count=int(analysis['small_tp_count']),
            fn_count=int(analysis['fn_count']),
        )
        self._vis_case_seq += 1

        self._vis_fp_candidates.append(candidate)
        self._vis_fp_candidates.sort(
            key=lambda x: (x['rank_score'], x['seq']), reverse=True)
        if len(self._vis_fp_candidates) > self.vis_fp_case_max:
            self._vis_fp_candidates = self._vis_fp_candidates[:self.vis_fp_case_max]

    def _analyze_weed_islands_for_visualization(self,
                                                pred_label: np.ndarray,
                                                gt_label: np.ndarray,
                                                instance_map: np.ndarray) -> Dict[str, object]:
        """Classify predicted weed islands for visualization (TP/FP/FN)."""
        class_label = self.class2_label

        gt_instance_ids = np.unique(instance_map[gt_label == class_label])
        gt_instance_ids = gt_instance_ids[gt_instance_ids > 0]
        gt_instance_ids = [int(x) for x in gt_instance_ids]

        pred_mask = pred_label == class_label
        pred_labeled, n_pred = sp_ndimage.label(pred_mask)

        comp_records = []
        gt_to_components: Dict[int, List[int]] = defaultdict(list)

        for comp_id in range(1, n_pred + 1):
            comp_mask = pred_labeled == comp_id
            if not comp_mask.any():
                continue

            comp_area = int(comp_mask.sum())
            if comp_area < self.pred_island_min_area:
                continue
            class_overlap_mask = np.logical_and(comp_mask, gt_label == class_label)
            overlapping_gt_ids = np.unique(instance_map[class_overlap_mask])
            overlapping_gt_ids = overlapping_gt_ids[overlapping_gt_ids > 0]
            overlapping_gt_ids = [int(x) for x in overlapping_gt_ids]

            is_fp_seed = len(overlapping_gt_ids) == 0
            rec = dict(
                comp_id=int(comp_id),
                area=comp_area,
                mask=comp_mask,
                overlap_gt_ids=overlapping_gt_ids,
                is_fp_seed=is_fp_seed,
                is_tp=False,
                is_fp=bool(is_fp_seed),
            )
            comp_records.append(rec)

            for gt_id in overlapping_gt_ids:
                gt_to_components[gt_id].append(int(comp_id))

        fn_masks = []
        tp_gt_ids = set()
        for gt_id in gt_instance_ids:
            gt_mask = instance_map == gt_id
            gt_area = int(gt_mask.sum())
            if gt_area <= 0:
                continue

            comp_ids = gt_to_components.get(gt_id, [])
            if len(comp_ids) == 0:
                fn_masks.append(dict(mask=gt_mask, area=gt_area))
                continue

            virtual_pred_mask = np.isin(pred_labeled, comp_ids)
            inter = int(np.logical_and(virtual_pred_mask, gt_mask).sum())
            iog = self._safe_div(inter, gt_area)

            if (not np.isnan(iog)) and (iog >= self.object_iog_thr):
                tp_gt_ids.add(int(gt_id))
            else:
                fn_masks.append(dict(mask=gt_mask, area=gt_area))

        # Exclusive component assignment: TP iff overlaps any TP GT instance.
        # Otherwise FP (either no overlap or overlaps only FN GT instances).
        for rec in comp_records:
            overlap_ids = rec['overlap_gt_ids']
            is_tp_comp = any(gt_id in tp_gt_ids for gt_id in overlap_ids)
            rec['is_tp'] = bool(is_tp_comp)
            rec['is_fp'] = not rec['is_tp']

        small_fp_count = 0
        small_tp_count = 0
        for rec in comp_records:
            if int(rec['area']) <= 100:
                if rec['is_fp']:
                    small_fp_count += 1
                elif rec['is_tp']:
                    small_tp_count += 1

        return dict(
            components=comp_records,
            fn_masks=fn_masks,
            fn_count=len(fn_masks),
            small_fp_count=small_fp_count,
            small_tp_count=small_tp_count,
        )

    def _dump_top_fp_visualizations(self, logger: MMLogger) -> None:
        if self.vis_output_dir is None or self.vis_fp_case_max <= 0:
            return
        if len(self._vis_fp_candidates) == 0:
            print_log('No FP-focused weed visualization cases selected.', logger=logger)
            return

        out_dir = osp.join(self.vis_output_dir, 'fp_focus_weed_small')
        mkdir_or_exist(out_dir)

        # Highest small FP count first.
        selected = sorted(
            self._vis_fp_candidates,
            key=lambda x: (x['rank_score'], x['seq']),
            reverse=True)[:self.vis_fp_case_max]

        for rank, item in enumerate(selected, start=1):
            vis_img = self._compose_fp_focus_visual(item)
            if vis_img is None:
                continue
            basename = osp.splitext(osp.basename(item['img_path']))[0]
            out_name = (
                f'{rank:02d}_{basename}_smallfp{int(item["small_fp_count"])}'
                f'_smalltp{int(item["small_tp_count"])}_fn{int(item["fn_count"])}.png')
            out_path = osp.join(out_dir, out_name)
            Image.fromarray(vis_img).save(out_path)

        print_log(
            f'Saved {len(selected)} FP-focused weed visualization images to: {out_dir}',
            logger=logger)

    def _compose_fp_focus_visual(self, item: dict) -> Optional[np.ndarray]:
        rgb = self._load_rgb_image(item['img_path'])
        gt_label = item['gt_label']
        pred_label = item['pred_label']
        if rgb.shape[:2] != gt_label.shape:
            rgb = self._resize_rgb_nearest(rgb, gt_label.shape)

        base = np.asarray(rgb, dtype=np.uint8)
        dim_bg = np.clip(base.astype(np.float32) * self.vis_bg_alpha, 0, 255).astype(np.uint8)

        # Left panel: GT semantic overlay on dim background.
        left = dim_bg.copy()
        crop_mask = gt_label == self.class1_label
        weed_mask = gt_label == self.class2_label
        crop_color = np.array([0, 255, 0], dtype=np.uint8)
        weed_color = np.array([255, 0, 0], dtype=np.uint8)
        if crop_mask.any():
            left[crop_mask] = crop_color
        if weed_mask.any():
            left[weed_mask] = weed_color

        # Right panel: model segmentation map + TP/FP/FN contour loops.
        right = self._render_prediction_segmentation(pred_label)
        tp_color = np.array([255, 165, 0], dtype=np.uint8)   # orange
        fp_color = np.array([0, 0, 255], dtype=np.uint8)     # blue
        fn_color = np.array([0, 128, 128], dtype=np.uint8)   # teal

        for rec in item['comp_records']:
            mask = rec['mask']
            if rec['is_tp']:
                right = self._draw_mask_loop(right, mask, tp_color)
            elif rec['is_fp']:
                right = self._draw_mask_loop(right, mask, fp_color)

        for fn_item in item['fn_masks']:
            right = self._draw_mask_loop(right, fn_item['mask'], fn_color)

        pair = np.concatenate([left, right], axis=1)
        pil_img = Image.fromarray(pair)
        draw = ImageDraw.Draw(pil_img)
        h, w = pair.shape[:2]
        mid_x = w // 2

        draw.line([(mid_x, 0), (mid_x, h - 1)], fill=(255, 255, 255), width=2)
        draw.rectangle([(0, 0), (360, 20)], fill=(0, 0, 0))
        draw.text((4, 2), 'GT (dim bg): crop=green, weed=red', fill=(255, 255, 255))
        draw.rectangle([(mid_x, 0), (min(mid_x + 520, w - 1), 20)], fill=(0, 0, 0))
        draw.text(
            (mid_x + 4, 2),
            'Pred segmentation + loops: TP=orange, FP=blue, FN=teal',
            fill=(255, 255, 255))

        footer = (
            f'smallFP<=100: {int(item["small_fp_count"])} | '
            f'smallTP<=100: {int(item["small_tp_count"])} | '
            f'FN(gt weeds): {int(item["fn_count"])} | '
            f'IoG>= {self.object_iog_thr * 100.0:.1f}% | '
            f'IoP>= {self.object_iop_thr * 100.0:.1f}%')
        draw.rectangle([(mid_x, 20), (min(mid_x + 620, w - 1), 40)], fill=(0, 0, 0))
        draw.text((mid_x + 4, 22), footer, fill=(255, 255, 255))

        # Draw per-segment labels with pixel areas on right panel.
        for idx, rec in enumerate(item['comp_records'], start=1):
            status = 'TP' if rec['is_tp'] else ('FP' if rec['is_fp'] else None)
            if status is None:
                continue
            ys, xs = np.where(rec['mask'])
            if ys.size == 0:
                continue
            x1, y1 = int(xs.min()) + mid_x, int(ys.min())
            label = f'#{idx} {status} A={int(rec["area"])}'
            ty = max(40, y1 - 12)
            draw.rectangle([(x1, ty), (min(x1 + 150, w - 1), ty + 12)], fill=(0, 0, 0))
            draw.text((x1 + 2, ty), label, fill=(255, 255, 255))

        for fn_idx, fn_item in enumerate(item['fn_masks'], start=1):
            ys, xs = np.where(fn_item['mask'])
            if ys.size == 0:
                continue
            x1, y1 = int(xs.min()) + mid_x, int(ys.min())
            label = f'FN#{fn_idx} A={int(fn_item["area"])}'
            ty = max(40, y1 - 12)
            draw.rectangle([(x1, ty), (min(x1 + 150, w - 1), ty + 12)], fill=(0, 0, 0))
            draw.text((x1 + 2, ty), label, fill=(255, 255, 255))

        return np.asarray(pil_img, dtype=np.uint8)

    def _render_prediction_segmentation(self, pred_label: np.ndarray) -> np.ndarray:
        """Render model semantic prediction as RGB map."""
        h, w = pred_label.shape
        seg = np.zeros((h, w, 3), dtype=np.uint8)
        # class0 background/soil -> black, class1 crop -> green, class2 weed -> red
        seg[pred_label == self.class1_label] = np.array([0, 255, 0], dtype=np.uint8)
        seg[pred_label == self.class2_label] = np.array([255, 0, 0], dtype=np.uint8)
        return seg

    @staticmethod
    def _draw_mask_loop(canvas: np.ndarray,
                        mask: np.ndarray,
                        color: np.ndarray) -> np.ndarray:
        """Draw a thin contour loop around a binary mask."""
        if not mask.any():
            return canvas
        eroded = sp_ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        boundary = np.logical_and(mask, np.logical_not(eroded))
        out = canvas.copy()
        out[boundary] = color
        return out

    def _resolve_img_path(self, data_sample: dict) -> Optional[str]:
        img_path = data_sample.get('img_path', None)
        if img_path is None and hasattr(data_sample, 'metainfo'):
            img_path = data_sample.metainfo.get('img_path', None)
        return img_path

    @staticmethod
    def _extract_date_bin(img_path: Optional[str]) -> str:
        """Extract MM-DD date prefix from filenames like '05-15_image.png'."""
        if img_path is None:
            return 'unknown'
        prefix = osp.splitext(osp.basename(img_path))[0].split('_')[0]
        if (len(prefix) == 5 and prefix[2] == '-'
                and prefix[:2].isdigit() and prefix[3:].isdigit()):
            return prefix
        return 'unknown'

    @staticmethod
    def _load_rgb_image(img_path: str) -> np.ndarray:
        return np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

    @staticmethod
    def _resize_rgb_nearest(rgb_image: np.ndarray,
                            target_shape: Tuple[int, int]) -> np.ndarray:
        dst_h, dst_w = target_shape
        pil_img = Image.fromarray(rgb_image)
        pil_img = pil_img.resize((dst_w, dst_h), resample=Image.BILINEAR)
        return np.array(pil_img, dtype=np.uint8)

    def _compute_object_level_stats(self,
                                    pred_label: np.ndarray,
                                    gt_label: np.ndarray,
                                    instance_map: np.ndarray,
                                    rgb: Optional[np.ndarray] = None) -> Dict[str, int]:
        """Compute IoG-recall and IoP-precision by class and size bins.

        Recall bins are defined by GT instance area and use IoG thresholding.
        Precision bins are defined by predicted component area and use IoP
        thresholding.
        """
        out = {'per_class': {}}
        out['fp_meta'] = {}
        if self.audit_thresholds is not None:
            out['raw_iog_iop'] = {}

        for class_label in self.object_eval_class_labels:
            class_out = {
                'all': self._new_count_bucket(),
                'le100': self._new_count_bucket(),
                'gt100': self._new_count_bucket(),
            }
            class_removed = {'all': 0.0, 'le100': 0.0, 'gt100': 0.0}
            class_fp_meta = {
                'all':   dict(edge_fp=0.0, dominant_counts=self._new_fp_dominant_counts(),
                              probably_unlabelled=0.0),
                'le100': dict(edge_fp=0.0, dominant_counts=self._new_fp_dominant_counts(),
                              probably_unlabelled=0.0),
                'gt100': dict(edge_fp=0.0, dominant_counts=self._new_fp_dominant_counts(),
                              probably_unlabelled=0.0),
            }
            if self.audit_thresholds is not None:
                _audit_gt: List[dict] = []

            gt_instance_ids = np.unique(instance_map[gt_label == class_label])
            gt_instance_ids = gt_instance_ids[gt_instance_ids > 0]
            gt_instance_ids = [int(x) for x in gt_instance_ids]

            gt_to_components: Dict[int, List[int]] = defaultdict(list)
            pred_mask = pred_label == class_label
            pred_labeled, n_pred = sp_ndimage.label(pred_mask)

            for comp_id in range(1, n_pred + 1):
                comp_mask = pred_labeled == comp_id
                if not comp_mask.any():
                    continue

                comp_area = int(comp_mask.sum())
                if comp_area < self.pred_island_min_area:
                    rem_bin = self._size_bin(comp_area)
                    class_removed['all'] += 1
                    class_removed[rem_bin] += 1
                    continue
                comp_bin = self._size_bin(comp_area)
                class_out['all']['pred_total'] += 1
                class_out[comp_bin]['pred_total'] += 1

                class_overlap_mask = np.logical_and(comp_mask, gt_label == class_label)
                overlapping_gt_ids = np.unique(instance_map[class_overlap_mask])
                overlapping_gt_ids = overlapping_gt_ids[overlapping_gt_ids > 0]

                for gt_id in overlapping_gt_ids:
                    gt_to_components[int(gt_id)].append(int(comp_id))

                # IoP precision decision for this predicted island.
                inter_class = int(class_overlap_mask.sum())
                iop = self._safe_div(inter_class, comp_area)

                # Accumulate raw pixel sums for per-pixel precision by size bin.
                class_out['all']['pixel_inter_sum'] += float(inter_class)
                class_out['all']['pixel_area_sum'] += float(comp_area)
                class_out[comp_bin]['pixel_inter_sum'] += float(inter_class)
                class_out[comp_bin]['pixel_area_sum'] += float(comp_area)

                is_precision_tp = (not np.isnan(iop)) and (iop >= self.object_iop_thr)

                if is_precision_tp:
                    class_out['all']['tp_precision'] += 1
                    class_out[comp_bin]['tp_precision'] += 1
                else:
                    class_out['all']['fp_precision'] += 1
                    class_out[comp_bin]['fp_precision'] += 1

                    dominant_gt_cat = self._dominant_gt_category(comp_mask, gt_label)
                    class_fp_meta['all']['dominant_counts'][dominant_gt_cat] += 1
                    class_fp_meta[comp_bin]['dominant_counts'][dominant_gt_cat] += 1
                    if self._touches_image_edge(comp_mask):
                        class_fp_meta['all']['edge_fp'] += 1
                        class_fp_meta[comp_bin]['edge_fp'] += 1

                    # Greenness check: FP-on-background with high ExG may be
                    # an unlabelled plant that the model correctly detected.
                    if (rgb is not None
                            and self.exg_unlabelled_thr is not None
                            and dominant_gt_cat == self._fp_cat_name(
                                self.class0_label)):
                        r = rgb[comp_mask, 0].astype(np.float32)
                        g = rgb[comp_mask, 1].astype(np.float32)
                        b = rgb[comp_mask, 2].astype(np.float32)
                        exg = float((2.0 * g - r - b).mean())
                        if exg > self.exg_unlabelled_thr:
                            class_fp_meta['all']['probably_unlabelled'] += 1
                            class_fp_meta[comp_bin]['probably_unlabelled'] += 1

            for gt_id in gt_instance_ids:
                gt_mask = instance_map == gt_id
                gt_area = int(gt_mask.sum())
                if gt_area <= 0:
                    continue

                gt_bin = self._size_bin(gt_area)
                class_out['all']['gt_total'] += 1
                class_out[gt_bin]['gt_total'] += 1

                comp_ids = gt_to_components.get(gt_id, [])
                if len(comp_ids) == 0:
                    class_out['all']['fn_recall'] += 1
                    class_out[gt_bin]['fn_recall'] += 1
                    if self.audit_thresholds is not None:
                        _audit_gt.append({'area': gt_area, 'iog': 0.0})
                    continue

                virtual_pred_mask = np.isin(pred_labeled, comp_ids)
                inter = int(np.logical_and(virtual_pred_mask, gt_mask).sum())
                iog = self._safe_div(inter, gt_area)
                if self.audit_thresholds is not None:
                    _audit_gt.append({
                        'area': gt_area,
                        'iog': 0.0 if np.isnan(iog) else float(iog),
                    })

                if (not np.isnan(iog)) and (iog >= self.object_iog_thr):
                    class_out['all']['tp_recall'] += 1
                    class_out[gt_bin]['tp_recall'] += 1
                else:
                    class_out['all']['fn_recall'] += 1
                    class_out[gt_bin]['fn_recall'] += 1

            for bin_key in ('all', 'le100', 'gt100'):
                bucket = class_out[bin_key]
                precision = self._safe_div(
                    bucket['tp_precision'], bucket['tp_precision'] + bucket['fp_precision'])
                recall = self._safe_div(
                    bucket['tp_recall'], bucket['tp_recall'] + bucket['fn_recall'])
                bucket['precision'] = precision
                bucket['recall'] = recall
                bucket['f1'] = self._f1(precision, recall)

            out['per_class'][class_label] = class_out
            if 'removed_small' not in out:
                out['removed_small'] = {}
            out['removed_small'][class_label] = class_removed
            out['fp_meta'][class_label] = class_fp_meta
            if self.audit_thresholds is not None:
                out['raw_iog_iop'][class_label] = {'gt': _audit_gt}

        return out

    def _compute_pixel_confusion_matrix(self,
                                        pred_label: np.ndarray,
                                        gt_label: np.ndarray) -> np.ndarray:
        """Compute 3x3 confusion matrix for [class0, class1, class2]."""
        conf = np.zeros((3, 3), dtype=np.int64)

        valid_mask = gt_label != self.ignore_index
        for ignored_label in self.ignore_label_ids:
            valid_mask &= (gt_label != ignored_label)

        label_to_index = {
            self.class0_label: 0,
            self.class1_label: 1,
            self.class2_label: 2,
        }

        gt_valid = gt_label[valid_mask]
        pred_valid = pred_label[valid_mask]

        in_eval = np.isin(gt_valid, self.eval_label_ids) & np.isin(
            pred_valid, self.eval_label_ids)
        gt_valid = gt_valid[in_eval]
        pred_valid = pred_valid[in_eval]

        for gt_id, pred_id in zip(gt_valid, pred_valid):
            conf[label_to_index[int(gt_id)], label_to_index[int(pred_id)]] += 1

        return conf

    def _derive_per_class_from_confusion(self,
                                         confusion: np.ndarray) -> List[Dict[str, float]]:
        out = []
        for idx in range(3):
            tp = float(confusion[idx, idx])
            fp = float(confusion[:, idx].sum() - confusion[idx, idx])
            fn = float(confusion[idx, :].sum() - confusion[idx, idx])
            iou = self._safe_div(tp, tp + fp + fn)
            precision = self._safe_div(tp, tp + fp)
            recall = self._safe_div(tp, tp + fn)
            f1 = self._f1(precision, recall)
            out.append(
                dict(
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    iou=iou,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                ))
        return out

    def _log_paper_summary(self,
                          logger: MMLogger,
                          metrics: Dict[str, float],
                          n_images: int) -> None:
        """Print the 9-column WACV paper metrics row for this model."""
        n = max(1, n_images)
        fp_soil_raw = metrics.get('obj_weed_le100_fp_on_background_soil', 0.0)
        fp_veg = metrics.get('obj_weed_le100_fp_probably_unlabelled', 0.0)
        fp_bare = fp_soil_raw - fp_veg
        fp_crop = metrics.get('obj_weed_le100_fp_on_crop', 0.0)

        table = PrettyTable()
        table.field_names = [
            'IoU Crop(%)',
            'IoU Weed(%)',
            'WeedPrec All(%)',
            'WeedPrec <=100px(%)',
            'TinyRec Crop(%)',
            'TinyRec Weed(%)',
            'FP/img Crop',
            'FP/img Soil_bare',
            'FP/img Soil_veg',
        ]
        table.add_row([
            metrics.get('pixel_IoU_crop', float('nan')),
            metrics.get('weed_iou', float('nan')),
            metrics.get('weed_precision', float('nan')),
            metrics.get('weed_pixel_prec_le100', float('nan')),
            metrics.get('obj_crop_le100_iog_recall', float('nan')),
            metrics.get('obj_weed_le100_iog_recall', float('nan')),
            round(fp_crop / n, 2),
            round(fp_bare / n, 2),
            round(fp_veg / n, 2),
        ])
        print_log('\nWACV Paper Table (N={:d} images):'.format(n_images), logger=logger)
        print_log('\n' + table.get_string(), logger=logger)

    def _log_object_summary(self,
                            logger: MMLogger,
                            object_totals: Dict[int, Dict[str, Dict[str, float]]],
                            pixel_metrics: List[Dict[str, float]],
                            fp_breakdown_totals: Dict[int, Dict[str, dict]]) -> None:
        table = PrettyTable()
        exg_col = (f'Prob.Unlabelled\n(ExG>{self.exg_unlabelled_thr:.0f})'
                   if self.exg_unlabelled_thr is not None else 'Prob.Unlabelled')
        table.field_names = [
            'Plant Category',
            'GT Instances',
            'Detected',
            'IoU',
            f'Recall (IoG>={self.object_iog_thr * 100.0:.1f}%)',
            f'Obj-Precision (IoP>={self.object_iop_thr * 100.0:.1f}%)',
            'Pixel-Precision',
            'F1',
            f'Pred on {self._fp_cat_name(self.class0_label)}',
            f'Pred on {self._fp_cat_name(self.class1_label)}',
            exg_col,
        ]

        row_specs = [
            (self.class1_label, 'le100', 'Tiny crop (<100px)'),
            (self.class1_label, 'gt100', 'Large crop (>=100px)'),
            (self.class2_label, 'le100', 'Tiny weed (<100px)'),
            (self.class2_label, 'gt100', 'Large weed (>=100px)'),
        ]

        for class_label, bin_key, row_name in row_specs:
            pixel_idx = 1 if class_label == self.class1_label else 2
            px = pixel_metrics[pixel_idx]
            obj = object_totals[class_label][bin_key]
            fp_bin = fp_breakdown_totals[class_label][bin_key]
            dom = fp_bin['dominant_counts']

            table.add_row([
                row_name,
                int(obj['gt_total']),
                int(obj['tp_recall']),
                np.round(px['iou'] * 100.0, 2),
                np.round(obj['recall'] * 100.0, 2),
                np.round(obj['precision'] * 100.0, 2),
                np.round(px['precision'] * 100.0, 2),
                np.round(obj['f1'] * 100.0, 2),
                int(dom[self._fp_cat_name(self.class0_label)]),
                int(dom[self._fp_cat_name(self.class1_label)]),
                int(fp_bin.get('probably_unlabelled', 0)),
            ])

        print_log(
            'single summary table (tiny/large crop & weed) with IoG/IoP '
            'object metrics and FP source counts:',
            logger=logger)
        print_log('\n' + table.get_string(), logger=logger)

    def _log_pixel_summary(self,
                           logger: MMLogger,
                           confusion: np.ndarray,
                           per_class: List[Dict[str, float]]) -> None:
        conf_table = PrettyTable()
        conf_table.field_names = [
            'GT\\Pred',
            self.class_names[0],
            self.class_names[1],
            self.class_names[2],
        ]
        for row_idx in range(3):
            conf_table.add_row([
                self.class_names[row_idx],
                int(confusion[row_idx, 0]),
                int(confusion[row_idx, 1]),
                int(confusion[row_idx, 2]),
            ])

        cls_table = PrettyTable()
        cls_table.field_names = ['Class', 'IoU', 'Precision', 'Recall', 'F1']
        for idx, cls_name in enumerate(self.class_names):
            cls_table.add_row([
                cls_name,
                np.round(per_class[idx]['iou'] * 100.0, 2),
                np.round(per_class[idx]['precision'] * 100.0, 2),
                np.round(per_class[idx]['recall'] * 100.0, 2),
                np.round(per_class[idx]['f1'] * 100.0, 2),
            ])

        print_log('pixel-level 3x3 confusion matrix:', logger=logger)
        print_log('\n' + conf_table.get_string(), logger=logger)
        print_log('pixel-level class metrics:', logger=logger)
        print_log('\n' + cls_table.get_string(), logger=logger)

    def _log_fp_breakdown_summary(self,
                                  logger: MMLogger,
                                  fp_breakdown_totals: Dict[int, Dict[str, dict]]) -> None:
        table = PrettyTable()
        table.field_names = [
            'Pred Class',
            'Size',
            'FP total',
            f'FP on {self._fp_cat_name(self.class0_label)}',
            f'FP on {self._fp_cat_name(self.class1_label)}',
            f'FP on {self._fp_cat_name(self.class2_label)}',
            'FP on ignore',
            'FP on other',
            'FP edge-truncated',
        ]

        for class_label in self.object_eval_class_labels:
            pred_name = self._class_name_for_label(class_label)
            for bin_key in ('all', 'le100', 'gt100'):
                size_name = 'all' if bin_key == 'all' else ('<=100' if bin_key == 'le100' else '>100')
                fp_bin = fp_breakdown_totals[class_label][bin_key]
                dom = fp_bin['dominant_counts']
                fp_total = int(
                    dom[self._fp_cat_name(self.class0_label)] +
                    dom[self._fp_cat_name(self.class1_label)] +
                    dom[self._fp_cat_name(self.class2_label)] +
                    dom['ignore'] + dom['other'])
                table.add_row([
                    pred_name,
                    size_name,
                    fp_total,
                    int(dom[self._fp_cat_name(self.class0_label)]),
                    int(dom[self._fp_cat_name(self.class1_label)]),
                    int(dom[self._fp_cat_name(self.class2_label)]),
                    int(dom['ignore']),
                    int(dom['other']),
                    int(fp_bin['edge_fp']),
                ])

        print_log('FP breakdown by GT source and edge truncation:', logger=logger)
        print_log('\n' + table.get_string(), logger=logger)

    def _log_date_breakdown_table(
            self, results: list, logger: MMLogger) -> None:
        """Log tiny-instance (<=100 px) recall per acquisition date.

        Date prefix is inferred from the image filename (MM-DD pattern before
        the first underscore, e.g. '05-15_image.png' → '05-15').  The table
        is suppressed when fewer than two distinct dates are found.
        """
        date_bins = sorted(set(r.get('date_bin', 'unknown') for r in results))
        if len(date_bins) < 2 or date_bins == ['unknown']:
            return

        date_stats: Dict[str, Dict[int, Dict[str, float]]] = {
            db: {cl: {'gt_total': 0.0, 'tp_recall': 0.0}
                 for cl in self.object_eval_class_labels}
            for db in date_bins
        }

        for item in results:
            db = item.get('date_bin', 'unknown')
            per_class = item['object_stats'].get('per_class', {})
            for cl in self.object_eval_class_labels:
                le100 = per_class.get(cl, {}).get('le100', {})
                date_stats[db][cl]['gt_total'] += float(
                    le100.get('gt_total', 0.0))
                date_stats[db][cl]['tp_recall'] += float(
                    le100.get('tp_recall', 0.0))

        table = PrettyTable()
        header: list = ['Date']
        for cl in self.object_eval_class_labels:
            cls_name = self._class_name_for_label(cl)
            header += [
                f'{cls_name} GT (<=100px)',
                f'{cls_name} Detected',
                f'{cls_name} Recall %',
            ]
        table.field_names = header

        for db in date_bins:
            row: list = [db]
            for cl in self.object_eval_class_labels:
                s = date_stats[db][cl]
                gt = int(s['gt_total'])
                tp = int(s['tp_recall'])
                recall = self._safe_div(tp, gt)
                row += [
                    gt,
                    tp,
                    '-' if np.isnan(recall) else f'{recall * 100.0:.1f}',
                ]
            table.add_row(row)

        print_log(
            'Tiny instance recall by acquisition date:', logger=logger)
        print_log('\n' + table.get_string(), logger=logger)

    def _run_threshold_audit(
            self,
            results: list,
            logger: MMLogger,
            pixel_metrics: List[Dict[str, float]],
            fp_breakdown_totals: Dict[int, Dict[str, dict]]) -> None:
        """Sweep IoG recall threshold τ for tiny (<=100px) instances only.

        Recall for both weed and crop tiny instances varies with τ.  Weed pixel
        precision and FP/image counts are derived from the confusion matrix /
        FP breakdown and are constant across rows.
        """
        n_images = max(1, len(results))
        weed_cl = self.class2_label
        crop_cl = self.class1_label

        # Collect tiny GT IoG scores (<=100px) for weed and crop.
        weed_gt: List[dict] = []
        crop_gt: List[dict] = []
        for item in results:
            raw = item['object_stats'].get('raw_iog_iop', {})
            weed_gt.extend(
                g for g in raw.get(weed_cl, {}).get('gt', [])
                if g['area'] <= 100)
            crop_gt.extend(
                g for g in raw.get(crop_cl, {}).get('gt', [])
                if g['area'] <= 100)

        # Constant reference values (do not depend on τ).
        weed_pix_prec = pixel_metrics[weed_cl]['precision']
        fp_weed_bin = fp_breakdown_totals[weed_cl]['le100']['dominant_counts']
        fp_on_crop = fp_weed_bin.get(
            self._fp_cat_name(self.class1_label), 0.0) / n_images
        fp_on_soil = fp_weed_bin.get(
            self._fp_cat_name(self.class0_label), 0.0) / n_images

        weed_total = len(weed_gt)
        crop_total = len(crop_gt)
        rows: List[dict] = []
        for thr in self.audit_thresholds:
            weed_tp = sum(1 for g in weed_gt if g['iog'] >= thr)
            crop_tp = sum(1 for g in crop_gt if g['iog'] >= thr)
            rows.append({
                'thr': thr,
                'weed_total': weed_total, 'weed_tp': weed_tp,
                'weed_recall': self._safe_div(weed_tp, weed_total),
                'crop_total': crop_total, 'crop_tp': crop_tp,
                'crop_recall': self._safe_div(crop_tp, crop_total),
                'weed_pix_prec': weed_pix_prec,
                'fp_on_crop': fp_on_crop,
                'fp_on_soil': fp_on_soil,
            })

        self._print_audit_table(rows, logger)
        if self.audit_figure_path is not None:
            self._plot_threshold_audit(rows, logger)

    def _print_audit_table(self, rows: List[dict], logger: MMLogger) -> None:
        crop_name = self._class_name_for_label(self.class1_label)
        weed_name = self._class_name_for_label(self.class2_label)
        table = PrettyTable()
        table.field_names = [
            'tau (%)',
            f'{weed_name} Det/GT',
            f'{weed_name} Recall %',
            f'{crop_name} Det/GT',
            f'{crop_name} Recall %',
            'Weed Pix-Prec % (const)',
            f'FP/img {crop_name} (const)',
            'FP/img soil (const)',
        ]

        for row in rows:
            weed_recall_str = (
                '-' if np.isnan(row['weed_recall'])
                else f'{row["weed_recall"] * 100.0:.1f}')
            crop_recall_str = (
                '-' if np.isnan(row['crop_recall'])
                else f'{row["crop_recall"] * 100.0:.1f}')
            pix_str = (
                '-' if np.isnan(row['weed_pix_prec'])
                else f'{row["weed_pix_prec"] * 100.0:.1f}')
            table.add_row([
                f'{row["thr"] * 100.0:.0f}%',
                f'{int(row["weed_tp"])} / {int(row["weed_total"])}',
                weed_recall_str,
                f'{int(row["crop_tp"])} / {int(row["crop_total"])}',
                crop_recall_str,
                pix_str,
                f'{row["fp_on_crop"]:.2f}',
                f'{row["fp_on_soil"]:.2f}',
            ])

        print_log(
            'Threshold audit (tiny <=100px instances only) — recall varies '
            'with τ; weed pixel precision and FP/img are constant (reference):',
            logger=logger)
        print_log('\n' + table.get_string(), logger=logger)

    def _plot_threshold_audit(
            self, rows: List[dict], logger: MMLogger) -> None:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print_log(
                'matplotlib not available; skipping threshold audit figure.',
                logger=logger)
            return

        thrs_pct = [r['thr'] * 100.0 for r in rows]
        weed_recalls = [
            float('nan') if np.isnan(r['weed_recall'])
            else r['weed_recall'] * 100.0 for r in rows]
        crop_recalls = [
            float('nan') if np.isnan(r['crop_recall'])
            else r['crop_recall'] * 100.0 for r in rows]
        pix_prec_const = (
            float('nan') if np.isnan(rows[0]['weed_pix_prec'])
            else rows[0]['weed_pix_prec'] * 100.0)
        fp_crop_const = rows[0]['fp_on_crop']
        fp_soil_const = rows[0]['fp_on_soil']

        weed_name = self._class_name_for_label(self.class2_label)
        crop_name = self._class_name_for_label(self.class1_label)
        baseline_pct = self.object_iog_thr * 100.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Left: tiny instance recall curves for weed and crop + constant pix-prec.
        ax1.plot(thrs_pct, weed_recalls, 'r-o', linewidth=2,
                 label=f'{weed_name} Recall (IoG>=τ, <=100px)')
        ax1.plot(thrs_pct, crop_recalls, 'g-s', linewidth=2,
                 label=f'{crop_name} Recall (IoG>=τ, <=100px)')
        if not np.isnan(pix_prec_const):
            ax1.axhline(pix_prec_const, color='r', linestyle='--', linewidth=1.5,
                        label=f'{weed_name} Pix-Prec ({pix_prec_const:.1f}%, const)')
        ax1.axvline(baseline_pct, color='gray', linestyle=':', linewidth=1,
                    label=f'Baseline τ={baseline_pct:.0f}%')
        ax1.set_xlabel('Threshold τ (%)')
        ax1.set_ylabel('Recall (%)')
        ax1.set_title('Tiny instance (<=100px) Recall vs τ')
        ax1.legend(fontsize=8)
        ax1.set_xticks(thrs_pct)
        ax1.set_xticklabels([f'{t:.0f}%' for t in thrs_pct])
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)

        # Right: constant FP/img reference lines — flat across τ.
        ax2.axhline(fp_crop_const, color='g', linestyle='--', linewidth=1.5,
                    label=f'FP on {crop_name} ({fp_crop_const:.2f}/img, const)')
        ax2.axhline(fp_soil_const, color='m', linestyle='--', linewidth=1.5,
                    label=f'FP on soil ({fp_soil_const:.2f}/img, const)')
        ax2.axvline(baseline_pct, color='gray', linestyle=':', linewidth=1,
                    label=f'Baseline τ={baseline_pct:.0f}%')
        ax2.set_xlabel('Threshold τ (%)')
        ax2.set_ylabel('FP / image')
        ax2.set_title('FP per Image (<=100px) — constant across τ')
        ax2.legend(fontsize=8)
        ax2.set_xticks(thrs_pct)
        ax2.set_xticklabels([f'{t:.0f}%' for t in thrs_pct])
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f'Threshold Audit: τ={baseline_pct:.0f}% baseline '
            f'(IoG={self.object_iog_thr * 100:.0f}%, '
            f'IoP={self.object_iop_thr * 100:.0f}%)')
        fig.tight_layout()

        out_dir = osp.dirname(self.audit_figure_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(self.audit_figure_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print_log(
            f'Threshold audit figure saved to: {self.audit_figure_path}',
            logger=logger)

    def _resolve_instance_map_path(self, data_sample: dict) -> str:
        direct_path = data_sample.get('instance_map_path', None)
        if direct_path is None and hasattr(data_sample, 'metainfo'):
            direct_path = data_sample.metainfo.get('instance_map_path', None)
        if direct_path is not None:
            return direct_path

        seg_map_path = data_sample.get('seg_map_path', None)
        if seg_map_path is None and hasattr(data_sample, 'metainfo'):
            seg_map_path = data_sample.metainfo.get('seg_map_path', None)

        if self.instance_map_path is not None:
            img_path = data_sample.get('img_path', None)
            if img_path is None and hasattr(data_sample, 'metainfo'):
                img_path = data_sample.metainfo.get('img_path', None)
            if img_path is not None:
                stem = osp.splitext(osp.basename(img_path))[0]
                return osp.join(self.instance_map_path, stem + self.instance_map_suffix)

        if seg_map_path is not None:
            candidate = self._guess_from_seg_map_path(seg_map_path)
            if candidate is not None:
                return candidate

        raise KeyError(
            'Cannot resolve instance_map_path. Add instance_map_path in '
            'metainfo or set metric.instance_map_path.')

    def _guess_from_seg_map_path(self, seg_map_path: str) -> Optional[str]:
        seg_no_ext = osp.splitext(seg_map_path)[0]
        candidates = []
        for subdir in self.instance_map_subdirs:
            candidates.append(
                seg_no_ext.replace('/semantics/', f'/{subdir}/') +
                self.instance_map_suffix)

        for candidate in candidates:
            if osp.exists(candidate):
                return candidate
        if len(candidates) > 0:
            return candidates[-1]
        return None

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

    def _semantic_to_instance_map(self, gt_label: np.ndarray) -> np.ndarray:
        """Create pseudo instance map from GT weed connected-components."""
        inst = np.zeros(gt_label.shape, dtype=np.int64)
        weed_mask = gt_label == self.weed_label
        labeled, n = sp_ndimage.label(weed_mask)
        if n <= 0:
            return inst
        for comp_id in range(1, n + 1):
            comp_mask = labeled == comp_id
            if comp_mask.any():
                inst[comp_mask] = comp_id
        return inst

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
    def _safe_div(num: float, den: float) -> float:
        if den <= 0:
            return float('nan')
        return float(num) / float(den)

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        if np.isnan(precision) or np.isnan(recall):
            return float('nan')
        den = precision + recall
        if den <= 0:
            return float('nan')
        return 2.0 * precision * recall / den
