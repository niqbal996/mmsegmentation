#!/usr/bin/env python3
"""Visualize predicted weed FP components on background or crop.

Single-model mode  (--checkpoint only):
  For each validation image that has FP_bg or FP_crop weed predictions,
  saves a 3-panel PNG: RGB | GT mask | Prediction + overlays.

Comparative mode  (--checkpoint + --checkpoint2):
  For each FP component found in EITHER model, saves a cropped 6-panel PNG
  (2 rows × 3 cols):
      Row 1 – Model A:  RGB crop | GT crop | Prediction + overlays
      Row 2 – Model B:  RGB crop | GT crop | Prediction + overlays
  FP instances from both models are merged by mask IoU to avoid duplicates.

Usage (comparative)
-------------------
python tools/visualize_weed_fp_on_bg_crop.py \\
    configs/wacv/deeplabv3plus_r50-d8_...py \\
    /path/to/model1.pth \\
    --checkpoint2 /path/to/model2.pth \\
    --name1 DeepLabV3+ --name2 SegFormer \\
    --output-dir ./tmp/compare \\
    --iop-thr 0.05 --min-area 10 --max-vis 500
"""
import argparse
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage as sp_ndimage

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import mkdir_or_exist

from mmseg.apis import init_model
from mmseg.registry import DATASETS


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
_PALETTE = {
    'bg':       np.array([40,  40,  40],  dtype=np.uint8),
    'crop':     np.array([0,   200, 0],   dtype=np.uint8),
    'weed':     np.array([200, 0,   0],   dtype=np.uint8),
    'ignore':   np.array([128, 128, 128], dtype=np.uint8),
    'tp_weed':  np.array([255, 165, 0],   dtype=np.uint8),
    'fp_bg':    np.array([255, 230, 0],   dtype=np.uint8),
    'fp_crop':  np.array([0,   220, 220], dtype=np.uint8),
    'fn_weed':  np.array([0,   140, 140], dtype=np.uint8),
}
_LEGEND = [
    ('TP weed',    _PALETTE['tp_weed']),
    ('FP on bg',   _PALETTE['fp_bg']),
    ('FP on crop', _PALETTE['fp_crop']),
    ('FN weed',    _PALETTE['fn_weed']),
]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else float('nan')


def _draw_contour(canvas: np.ndarray, mask: np.ndarray,
                  color: np.ndarray, thickness: int = 2) -> np.ndarray:
    if not mask.any():
        return canvas
    eroded = sp_ndimage.binary_erosion(
        mask, structure=np.ones((3, 3), dtype=bool), iterations=thickness)
    out = canvas.copy()
    out[mask & ~eroded] = color
    return out


def _fill_overlay(canvas: np.ndarray, mask: np.ndarray,
                  color: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = canvas.astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + color.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_gt_mask(gt: np.ndarray, c0: int, c1: int, c2: int,
                    ignore_index: int = 255) -> np.ndarray:
    h, w = gt.shape
    rgb = np.full((h, w, 3), _PALETTE['bg'], dtype=np.uint8)
    rgb[gt == c1] = _PALETTE['crop']
    rgb[gt == c2] = _PALETTE['weed']
    rgb[gt == ignore_index] = _PALETTE['ignore']
    return rgb


def _render_pred_mask(pred: np.ndarray, c0: int, c1: int, c2: int) -> np.ndarray:
    h, w = pred.shape
    rgb = np.full((h, w, 3), _PALETTE['bg'], dtype=np.uint8)
    rgb[pred == c1] = _PALETTE['crop']
    rgb[pred == c2] = _PALETTE['weed']
    return rgb


def _make_legend_strip(height: int, entries,
                       swatch: int = 14, gap: int = 6,
                       pad: int = 10, strip_w: int = 130) -> np.ndarray:
    """Return a (height, strip_w, 3) uint8 array with a vertical legend."""
    strip = np.full((height, strip_w, 3), 20, dtype=np.uint8)
    pil   = Image.fromarray(strip)
    draw  = ImageDraw.Draw(pil)
    # Title
    draw.text((pad, pad), 'Legend', fill=(200, 200, 200))
    y = pad + 16
    for label, color in entries:
        c = tuple(int(v) for v in color)
        draw.rectangle([(pad, y), (pad + swatch, y + swatch)],
                       fill=c, outline=(80, 80, 80))
        draw.text((pad + swatch + 5, y + 1), label, fill=(230, 230, 230))
        y += swatch + gap
    return np.asarray(pil, dtype=np.uint8)


def _tight_text(draw: ImageDraw.Draw, x: int, y: int, text: str,
                fill, total_w: int, pad: int = 2) -> None:
    """Draw text with a background rectangle sized tightly to the text."""
    tb = draw.textbbox((x, y), text)
    draw.rectangle(
        [(tb[0] - pad, tb[1] - pad),
         (min(tb[2] + pad, total_w - 1), tb[3] + pad)],
        fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill)


# ---------------------------------------------------------------------------
# FP analysis
# ---------------------------------------------------------------------------

def analyse_weed_fps(pred_np: np.ndarray, gt_np: np.ndarray,
                     c0: int, c1: int, c2: int, ignore_index: int,
                     iop_thr: float, iog_thr: float, min_area: int) -> dict:
    pred_mask = pred_np == c2
    pred_labeled, n_pred = sp_ndimage.label(pred_mask)
    gt_weed = gt_np == c2
    gt_labeled, n_gt = sp_ndimage.label(gt_weed)

    gt_to_comps: Dict[int, List[int]] = defaultdict(list)
    components = []

    for cid in range(1, n_pred + 1):
        cm = pred_labeled == cid
        area = int(cm.sum())
        if area < min_area:
            continue

        iop = _safe_div(int((cm & gt_weed).sum()), area)
        is_tp_iop = (not np.isnan(iop)) and (iop >= iop_thr)

        for gid in (int(x) for x in np.unique(gt_labeled[cm & gt_weed]) if x > 0):
            gt_to_comps[gid].append(cid)

        gt_under = gt_np[cm]
        dom = int(gt_under[np.argmax(np.bincount(gt_under))])  if gt_under.size > 0 else -1

        if dom == c0:             cat = 'fp_bg'
        elif dom == c1:           cat = 'fp_crop'
        elif dom == c2:           cat = 'tp_weed'
        elif dom == ignore_index: cat = 'ignore'
        else:                     cat = 'other'

        components.append(dict(comp_id=cid, area=area, mask=cm,
                               iop=iop, is_tp_iop=is_tp_iop, category=cat))

    fn_masks = []
    tp_gt_ids = set()
    for gid in range(1, n_gt + 1):
        gm = gt_labeled == gid
        ga = int(gm.sum())
        if ga <= 0:
            continue
        cids = gt_to_comps.get(gid, [])
        if not cids:
            fn_masks.append(dict(mask=gm, area=ga))
            continue
        virtual = np.isin(pred_labeled, cids)
        iog = _safe_div(int((virtual & gm).sum()), ga)
        if (not np.isnan(iog)) and (iog >= iog_thr):
            tp_gt_ids.add(gid)
        else:
            fn_masks.append(dict(mask=gm, area=ga))

    for rec in components:
        if not rec['is_tp_iop']:
            rec['is_fp'], rec['is_tp'] = True, False
        else:
            rec['is_fp'], rec['is_tp'] = False, True
            rec['category'] = 'tp_weed'

    fp_bg   = sum(1 for r in components if r['is_fp'] and r['category'] == 'fp_bg')
    fp_crop = sum(1 for r in components if r['is_fp'] and r['category'] == 'fp_crop')
    return dict(components=components, fn_masks=fn_masks,
                has_fp_bg=fp_bg > 0, has_fp_crop=fp_crop > 0,
                fp_bg_count=fp_bg, fp_crop_count=fp_crop)


# ---------------------------------------------------------------------------
# Dataset / model helpers
# ---------------------------------------------------------------------------

def build_val_dataset(cfg: Config):
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    val_dl = cfg.get('val_dataloader', None)
    if val_dl is None:
        raise RuntimeError('Config has no val_dataloader.')
    ds = DATASETS.build(val_dl['dataset'])
    ds.full_init()
    return ds


def load_rgb(path: str, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    if target_shape is not None:
        h, w = target_shape
        img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_sample(dataset, idx: int):
    """Run dataset pipeline. Returns (data, gt_np, img_path) or None."""
    info = dataset.get_data_info(idx)
    img_path = info.get('img_path')
    seg_path = info.get('seg_map_path')
    if not img_path or not osp.exists(img_path):
        return None
    if not seg_path or not osp.exists(seg_path):
        return None
    try:
        data = dataset[idx]
    except Exception as e:
        print(f'  [idx={idx}] pipeline error: {e}')
        return None
    gt_np = data['data_samples'].gt_sem_seg.data.squeeze().cpu().numpy().astype(np.int64)
    return data, gt_np, img_path


def run_model(model, data: dict, gt_np: np.ndarray) -> np.ndarray:
    """Test-step one sample, return pred_np aligned to gt_np shape."""
    with torch.no_grad():
        results = model.test_step(
            {'inputs': [data['inputs']], 'data_samples': [data['data_samples']]})
    pred = results[0].pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int64)
    if pred.shape != gt_np.shape:
        pred = F.interpolate(
            torch.from_numpy(pred)[None, None].float(),
            size=gt_np.shape, mode='nearest')[0, 0].numpy().astype(np.int64)
    return pred


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------

def render_full_panels(rgb: np.ndarray, gt_np: np.ndarray, pred_np: np.ndarray,
                       analysis: dict, c0: int, c1: int, c2: int,
                       ignore_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = gt_np.shape
    panel_rgb  = np.array(Image.fromarray(rgb).resize((w, h), Image.BILINEAR), dtype=np.uint8)
    panel_gt   = _render_gt_mask(gt_np, c0, c1, c2, ignore_index)
    panel_pred = _render_pred_mask(pred_np, c0, c1, c2)

    for rec in analysis['components']:
        if rec['is_tp']:
            panel_pred = _draw_contour(panel_pred, rec['mask'], _PALETTE['tp_weed'])
        elif rec['is_fp']:
            col = _PALETTE.get(rec['category'], _PALETTE['fp_bg'])
            panel_pred = _fill_overlay(panel_pred, rec['mask'], col, alpha=0.5)
            panel_pred = _draw_contour(panel_pred, rec['mask'], col)
    for fn in analysis['fn_masks']:
        panel_pred = _draw_contour(panel_pred, fn['mask'], _PALETTE['fn_weed'])

    return panel_rgb, panel_gt, panel_pred


# ---------------------------------------------------------------------------
# Single-model full-image visualization
# ---------------------------------------------------------------------------

def compose_single_vis(rgb, gt_np, pred_np, analysis, c0, c1, c2,
                       ignore_index, iop_thr, iog_thr) -> np.ndarray:
    h, w = gt_np.shape
    pr, pg, pp = render_full_panels(rgb, gt_np, pred_np, analysis, c0, c1, c2, ignore_index)
    canvas = np.concatenate([pr, pg, pp], axis=1)
    pil    = Image.fromarray(canvas)
    draw   = ImageDraw.Draw(pil)
    W      = w * 3

    for x, title in [(0, 'RGB image'), (w, 'GT mask'),
                     (w * 2, 'Prediction + component labels')]:
        draw.rectangle([(x, 0), (x + w - 1, 18)], fill=(0, 0, 0))
        draw.text((x + 4, 2), title, fill=(255, 255, 255))
    draw.line([(w, 0), (w, h - 1)],     fill=(200, 200, 200), width=1)
    draw.line([(w*2, 0), (w*2, h - 1)], fill=(200, 200, 200), width=1)

    for i, rec in enumerate(analysis['components'], 1):
        if not (rec['is_tp'] or rec['is_fp']):
            continue
        ys, xs = np.where(rec['mask'])
        if ys.size == 0:
            continue
        tag = f"#{i} {'TP' if rec['is_tp'] else rec['category'].upper()} A={rec['area']}"
        _tight_text(draw,
                    min(int(xs.min()) + w * 2, W - 80),
                    max(20, int(ys.min()) - 12),
                    tag, (255, 255, 255), W)

    for i, fn in enumerate(analysis['fn_masks'], 1):
        ys, xs = np.where(fn['mask'])
        if ys.size == 0:
            continue
        _tight_text(draw,
                    min(int(xs.min()) + w * 2, W - 80),
                    max(20, int(ys.min()) - 12),
                    f"FN#{i} A={fn['area']}",
                    tuple(_PALETTE['fn_weed'].tolist()), W)

    summary = (f"FP-on-bg: {analysis['fp_bg_count']}  "
               f"FP-on-crop: {analysis['fp_crop_count']}  "
               f"FN-weed: {len(analysis['fn_masks'])}  "
               f"IoP>={iop_thr*100:.1f}%  IoG>={iog_thr*100:.1f}%")
    draw.rectangle([(0, h - 16), (W - 1, h - 1)], fill=(0, 0, 0))
    draw.text((4, h - 14), summary, fill=(255, 255, 200))
    panels = np.asarray(pil, dtype=np.uint8)
    legend = _make_legend_strip(h, _LEGEND)
    return np.concatenate([panels, legend], axis=1)


# ---------------------------------------------------------------------------
# Comparative crop visualization
# ---------------------------------------------------------------------------

def compute_crop_bbox(mask: np.ndarray, h: int, w: int,
                      pad_factor: float = 0.6, min_pad: int = 50):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, h, w
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    pad = max(min_pad, int(max(y2 - y1 + 1, x2 - x1 + 1) * pad_factor))
    return max(0, y1 - pad), max(0, x1 - pad), min(h, y2 + pad + 1), min(w, x2 + pad + 1)


def _c(arr: np.ndarray, cy1, cx1, cy2, cx2) -> np.ndarray:
    return arr[cy1:cy2, cx1:cx2] if arr.ndim == 2 else arr[cy1:cy2, cx1:cx2, :]


def _comp_labels_in_crop(draw, analysis, cy1, cx1, row_offset,
                          cw, total_w, text_fill=(255, 255, 255)):
    """Draw tight component labels for components visible inside a crop panel."""
    for i, rec in enumerate(analysis['components'], 1):
        if not (rec['is_tp'] or rec['is_fp']):
            continue
        ys, xs = np.where(rec['mask'])
        if ys.size == 0:
            continue
        if ys.max() < cy1 or ys.min() > (cy1 + draw.im.size[1] // 2 - row_offset):
            continue
        tag = f"#{i} {'TP' if rec['is_tp'] else rec['category'].upper()} A={rec['area']}"
        ty  = row_offset + max(14, int(ys.min()) - cy1 - 12)
        tx  = min(int(xs.min()) - cx1 + cw * 2, total_w - 80)
        _tight_text(draw, tx, ty, tag, text_fill, total_w)


def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = int((m1 & m2).sum())
    union = int((m1 | m2).sum())
    return inter / union if union > 0 else 0.0


def merge_fp_instances(analysis1: dict, analysis2: dict,
                       iou_thr: float = 0.4) -> List[dict]:
    """Union FP components from both models, merging overlapping ones.

    Returns list of dicts: {source, category, mask, area, rec_m1, rec_m2}
    """
    fps1 = [r for r in analysis1['components']
            if r['is_fp'] and r['category'] in ('fp_bg', 'fp_crop')]
    fps2 = [r for r in analysis2['components']
            if r['is_fp'] and r['category'] in ('fp_bg', 'fp_crop')]

    matched2 = set()
    instances = []

    for r1 in fps1:
        best_iou, best_j = 0.0, -1
        for j, r2 in enumerate(fps2):
            iou = _mask_iou(r1['mask'], r2['mask'])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            matched2.add(best_j)
            r2 = fps2[best_j]
            combined = r1['mask'] | r2['mask']
            instances.append(dict(source='both', category=r1['category'],
                                  mask=combined, area=int(combined.sum()),
                                  rec_m1=r1, rec_m2=r2))
        else:
            instances.append(dict(source='m1', category=r1['category'],
                                  mask=r1['mask'], area=r1['area'],
                                  rec_m1=r1, rec_m2=None))

    for j, r2 in enumerate(fps2):
        if j not in matched2:
            instances.append(dict(source='m2', category=r2['category'],
                                  mask=r2['mask'], area=r2['area'],
                                  rec_m1=None, rec_m2=r2))
    return instances


def compose_comparison_crop(rgb, gt_np, pred1_np, analysis1,
                             pred2_np, analysis2,
                             cy1, cx1, cy2, cx2,
                             c0, c1, c2, ignore_index,
                             name1, name2, fp_inst,
                             iop_thr, iog_thr) -> np.ndarray:
    """2-row × 3-col crop comparison image."""
    pr, pg, pp1 = render_full_panels(rgb, gt_np, pred1_np, analysis1, c0, c1, c2, ignore_index)
    _,   _,  pp2 = render_full_panels(rgb, gt_np, pred2_np, analysis2, c0, c1, c2, ignore_index)

    row1   = np.concatenate([_c(pr, cy1, cx1, cy2, cx2),
                              _c(pg, cy1, cx1, cy2, cx2),
                              _c(pp1, cy1, cx1, cy2, cx2)], axis=1)
    row2   = np.concatenate([_c(pr, cy1, cx1, cy2, cx2),
                              _c(pg, cy1, cx1, cy2, cx2),
                              _c(pp2, cy1, cx1, cy2, cx2)], axis=1)
    canvas = np.concatenate([row1, row2], axis=0)

    pil    = Image.fromarray(canvas)
    draw   = ImageDraw.Draw(pil)
    cw, ch = cx2 - cx1, cy2 - cy1
    W      = cw * 3

    # Dividers
    for col in (cw, cw * 2):
        draw.line([(col, 0), (col, ch * 2 - 1)], fill=(160, 160, 160), width=1)
    draw.line([(0, ch), (W - 1, ch)], fill=(255, 255, 255), width=2)

    # Row headers
    draw.rectangle([(0, 0), (W - 1, 13)], fill=(0, 0, 0))
    draw.text((4, 2),      'RGB',       fill=(200, 200, 200))
    draw.text((cw + 4, 2), 'GT mask',   fill=(200, 200, 200))
    draw.text((cw*2 + 4, 2), f'▲ {name1}', fill=(255, 220, 80))

    draw.rectangle([(0, ch), (W - 1, ch + 13)], fill=(0, 0, 0))
    draw.text((4, ch + 2),      'RGB',      fill=(200, 200, 200))
    draw.text((cw + 4, ch + 2), 'GT mask',  fill=(200, 200, 200))
    draw.text((cw*2 + 4, ch + 2), f'▼ {name2}', fill=(80, 200, 255))

    # Component labels per row (only if visible in crop)
    _comp_labels_in_crop(draw, analysis1, cy1, cx1, row_offset=0,  cw=cw, total_w=W)
    _comp_labels_in_crop(draw, analysis2, cy1, cx1, row_offset=ch, cw=cw, total_w=W)

    # Footer
    m1 = f'M1 fpbg={analysis1["fp_bg_count"]} fpcrop={analysis1["fp_crop_count"]} fn={len(analysis1["fn_masks"])}'
    m2 = f'M2 fpbg={analysis2["fp_bg_count"]} fpcrop={analysis2["fp_crop_count"]} fn={len(analysis2["fn_masks"])}'
    footer = (f'src:{fp_inst["source"]}  cat:{fp_inst["category"]}  '
              f'a={fp_inst["area"]}  |  {m1}  |  {m2}  |  '
              f'IoP>={iop_thr*100:.0f}% IoG>={iog_thr*100:.0f}%')
    draw.rectangle([(0, ch*2 - 15), (W - 1, ch*2 - 1)], fill=(0, 0, 0))
    draw.text((4, ch*2 - 13), footer, fill=(255, 255, 200))

    panels = np.asarray(pil, dtype=np.uint8)
    legend = _make_legend_strip(ch * 2, _LEGEND)
    return np.concatenate([panels, legend], axis=1)


# ---------------------------------------------------------------------------
# Argument parsing + label-id extraction
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--checkpoint2',  default=None,
                   help='Second checkpoint — enables comparative crop mode')
    p.add_argument('--config2',      default=None,
                   help='Config for model 2 (defaults to same as model 1)')
    p.add_argument('--name1',        default=None)
    p.add_argument('--name2',        default=None)
    p.add_argument('--output-dir',   required=True)
    p.add_argument('--iop-thr',      type=float, default=0.05)
    p.add_argument('--iog-thr',      type=float, default=0.05)
    p.add_argument('--min-area',     type=int,   default=0)
    p.add_argument('--max-vis',      type=int,   default=500)
    p.add_argument('--device',       default='cuda:0')
    p.add_argument('--fp-bg-only',   action='store_true')
    p.add_argument('--fp-crop-only', action='store_true')
    p.add_argument('--pad-factor',   type=float, default=0.6,
                   help='Padding around FP component as fraction of its size')
    p.add_argument('--min-pad',      type=int,   default=50,
                   help='Minimum padding in pixels')
    p.add_argument('--dedup-iou',    type=float, default=0.4,
                   help='Mask IoU threshold for merging FPs across models')
    p.add_argument('--class0-label', type=int,   default=None)
    p.add_argument('--class1-label', type=int,   default=None)
    p.add_argument('--class2-label', type=int,   default=None)
    p.add_argument('--ignore-index', type=int,   default=255)
    return p.parse_args()


def _label_ids_from_cfg(cfg, d0=0, d1=1, d2=2):
    def _from(d):
        return int(d.get('class0_label', d0)), int(d.get('class1_label', d1)), int(d.get('class2_label', d2))
    ev = cfg.get('val_evaluator', None)
    if ev is None:
        return d0, d1, d2
    if isinstance(ev, dict):
        if ev.get('type') == 'InstanceDetectionMetric':
            return _from(ev)
        for m in ev.get('metrics', []):
            if isinstance(m, dict) and m.get('type') == 'InstanceDetectionMetric':
                return _from(m)
    elif isinstance(ev, list):
        for m in ev:
            if isinstance(m, dict) and m.get('type') == 'InstanceDetectionMetric':
                return _from(m)
    return d0, d1, d2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    mkdir_or_exist(args.output_dir)

    cfg = Config.fromfile(args.config)
    c0, c1, c2 = _label_ids_from_cfg(cfg)
    if args.class0_label is not None: c0 = args.class0_label
    if args.class1_label is not None: c1 = args.class1_label
    if args.class2_label is not None: c2 = args.class2_label
    ignore_index = args.ignore_index
    print(f'Labels  bg={c0}  crop={c1}  weed={c2}  ignore={ignore_index}')
    print(f'IoP>={args.iop_thr*100:.1f}%  IoG>={args.iog_thr*100:.1f}%  min_area={args.min_area}')

    dataset = build_val_dataset(cfg)
    print(f'{len(dataset)} validation samples.')

    model1 = init_model(args.config, args.checkpoint, device=args.device)
    model1.eval()

    comparative = args.checkpoint2 is not None
    if comparative:
        model2 = init_model(args.config2 or args.config, args.checkpoint2, device=args.device)
        model2.eval()
        name1 = args.name1 or osp.splitext(osp.basename(args.checkpoint))[0]
        name2 = args.name2 or osp.splitext(osp.basename(args.checkpoint2))[0]
        print(f'Comparative mode:  M1={name1}   M2={name2}')
    else:
        print('Single-model mode.')

    saved = 0

    for idx in range(len(dataset)):
        if saved >= args.max_vis:
            print(f'Reached --max-vis={args.max_vis}.')
            break

        sample = load_sample(dataset, idx)
        if sample is None:
            continue
        data, gt_np, img_path = sample
        h, w = gt_np.shape

        pred1 = run_model(model1, data, gt_np)
        an1   = analyse_weed_fps(pred1, gt_np, c0, c1, c2, ignore_index,
                                  args.iop_thr, args.iog_thr, args.min_area)

        # ── Single-model mode ──────────────────────────────────────────────
        if not comparative:
            if args.fp_bg_only   and not an1['has_fp_bg']:   continue
            if args.fp_crop_only and not an1['has_fp_crop']: continue
            if not an1['has_fp_bg'] and not an1['has_fp_crop']: continue

            rgb = load_rgb(img_path, target_shape=(h, w))
            vis = compose_single_vis(rgb, gt_np, pred1, an1, c0, c1, c2,
                                     ignore_index, args.iop_thr, args.iog_thr)
            stem = osp.splitext(osp.basename(img_path))[0]
            out  = osp.join(args.output_dir,
                            f'{saved+1:04d}_{stem}'
                            f'_fpbg{an1["fp_bg_count"]}'
                            f'_fpcrop{an1["fp_crop_count"]}'
                            f'_fn{len(an1["fn_masks"])}.png')
            Image.fromarray(vis).save(out)
            saved += 1
            print(f'[{idx:04d}] {saved} → {osp.basename(out)}')
            continue

        # ── Comparative mode ───────────────────────────────────────────────
        pred2 = run_model(model2, data, gt_np)
        an2   = analyse_weed_fps(pred2, gt_np, c0, c1, c2, ignore_index,
                                  args.iop_thr, args.iog_thr, args.min_area)

        if not any([an1['has_fp_bg'], an1['has_fp_crop'],
                    an2['has_fp_bg'], an2['has_fp_crop']]):
            continue

        rgb         = load_rgb(img_path, target_shape=(h, w))
        fp_instances = merge_fp_instances(an1, an2, iou_thr=args.dedup_iou)

        if args.fp_bg_only:
            fp_instances = [f for f in fp_instances if f['category'] == 'fp_bg']
        if args.fp_crop_only:
            fp_instances = [f for f in fp_instances if f['category'] == 'fp_crop']

        stem = osp.splitext(osp.basename(img_path))[0]
        for fp in fp_instances:
            if saved >= args.max_vis:
                break
            cy1, cx1, cy2, cx2 = compute_crop_bbox(
                fp['mask'], h, w, args.pad_factor, args.min_pad)
            vis = compose_comparison_crop(
                rgb, gt_np, pred1, an1, pred2, an2,
                cy1, cx1, cy2, cx2,
                c0, c1, c2, ignore_index,
                name1, name2, fp, args.iop_thr, args.iog_thr)
            out = osp.join(args.output_dir,
                           f'{saved+1:04d}_{stem}'
                           f'_src{fp["source"]}'
                           f'_{fp["category"]}'
                           f'_a{fp["area"]}.png')
            Image.fromarray(vis).save(out)
            saved += 1
            print(f'  [{idx:04d}] {saved}: {osp.basename(out)}')

    print(f'\nDone. {saved} images saved to {args.output_dir}')


if __name__ == '__main__':
    main()
