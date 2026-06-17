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


def _compute_exg(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Mean Excess Green (2G - R - B) for pixels under mask.

    Positive ExG (especially >30) indicates green vegetation.
    Near-zero or negative values suggest bare soil / background.
    """
    if not mask.any():
        return float('nan')
    r = rgb[mask, 0].astype(np.float32)
    g = rgb[mask, 1].astype(np.float32)
    b = rgb[mask, 2].astype(np.float32)
    return float((2.0 * g - r - b).mean())


def _exg_color(exg: float):
    """Traffic-light colour for an ExG value."""
    if np.isnan(exg):      return (100, 100, 100)
    if exg > 30:           return (60,  210, 60)   # green  — likely vegetation
    if exg > 10:           return (220, 180, 40)   # yellow — possibly vegetation
    return                        (130, 130, 130)   # gray   — likely soil/bg


def _make_legend_left(height: int, strip_w: int = 118) -> np.ndarray:
    """Narrow left sidebar with colour legend and ExG traffic-light key."""
    strip = np.full((height, strip_w, 3), 20, dtype=np.uint8)
    pil   = Image.fromarray(strip)
    draw  = ImageDraw.Draw(pil)
    pad, y = 8, 8
    draw.text((pad, y), 'Legend', fill=(200, 200, 200))
    y += 14
    sw, gap = 12, 4
    for lbl, color in _LEGEND:
        c = tuple(int(v) for v in color)
        draw.rectangle([(pad, y), (pad + sw, y + sw)], fill=c, outline=(60, 60, 60))
        draw.text((pad + sw + 4, y + 1), lbl, fill=(215, 215, 215))
        y += sw + gap
    y += 5
    draw.line([(pad, y), (strip_w - pad, y)], fill=(55, 55, 55), width=1)
    y += 7
    draw.text((pad, y), 'ExG=2G-R-B', fill=(95, 95, 95))
    y += 12
    for dot_c, label in [
            ((60,  210, 60),  '>30  veg'),
            ((220, 180, 40),  '>10  maybe'),
            ((130, 130, 130), 'else  soil'),
    ]:
        draw.ellipse([(pad, y + 1), (pad + 8, y + 9)],
                     fill=dot_c, outline=(50, 50, 50))
        draw.text((pad + 11, y), label, fill=(160, 160, 160))
        y += 12
    return np.asarray(pil, dtype=np.uint8)


def _make_info_strip(height: int,
                     components: list, fn_masks: list,
                     rgb_panel: np.ndarray,
                     model_name: str = '',
                     cy1: int = 0, cx1: int = 0,
                     cy2: Optional[int] = None, cx2: Optional[int] = None,
                     strip_w: int = 170) -> np.ndarray:
    """Per-row right sidebar: model header + instances visible in the crop region.

    cy1/cx1/cy2/cx2: crop bounds on the full-image masks.  Pass None to show all.
    """
    strip = np.full((height, strip_w, 3), 20, dtype=np.uint8)
    pil   = Image.fromarray(strip)
    draw  = ImageDraw.Draw(pil)
    pad, y = 8, 8

    if model_name:
        draw.text((pad, y), model_name, fill=(170, 195, 220))
        y += 14
        draw.line([(pad, y), (strip_w - pad, y)], fill=(55, 55, 55), width=1)
        y += 6

    def _in_crop(mask: np.ndarray) -> bool:
        if cy2 is None or cx2 is None:
            return True
        return bool(mask[cy1:cy2, cx1:cx2].any())

    for i, rec in enumerate(components, 1):
        if y + 12 > height:
            break
        if not (rec['is_tp'] or rec['is_fp']):
            continue
        if not _in_crop(rec['mask']):
            continue
        cat_s = {'tp_weed': 'TP', 'fp_bg': 'FP-bg', 'fp_crop': 'FP-cr'}.get(
            rec['category'], '?')
        if rec['is_fp'] and rec['category'] == 'fp_bg':
            exg   = _compute_exg(rgb_panel, rec['mask'])
            dot_c = _exg_color(exg)
            draw.ellipse([(pad, y + 2), (pad + 8, y + 10)],
                         fill=dot_c, outline=(50, 50, 50))
            draw.text((pad + 11, y),
                      f'#{i} {cat_s}  A={rec["area"]}  {exg:+.0f}',
                      fill=(210, 210, 210))
        else:
            draw.text((pad, y),
                      f'#{i} {cat_s}  A={rec["area"]}',
                      fill=(195, 195, 195))
        y += 12

    teal = tuple(_PALETTE['fn_weed'].tolist())
    for i, fn in enumerate(fn_masks, 1):
        if y + 12 > height:
            break
        if not _in_crop(fn['mask']):
            continue
        draw.text((pad, y), f'FN#{i}  A={fn["area"]}', fill=teal)
        y += 12

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

    for x, title in [(0, 'RGB'), (w, 'GT mask'), (w * 2, 'Prediction')]:
        draw.rectangle([(x, 0), (x + w - 1, 14)], fill=(0, 0, 0))
        draw.text((x + 4, 2), title, fill=(255, 255, 255))
    draw.line([(w, 0),   (w,   h - 1)], fill=(160, 160, 160), width=1)
    draw.line([(w*2, 0), (w*2, h - 1)], fill=(160, 160, 160), width=1)

    _draw_pred_labels(draw, analysis, cy1=0, cx1=0, ch=h, cw=w)

    panels = np.asarray(pil, dtype=np.uint8)
    strip  = _make_info_strip(h, analysis['components'], analysis['fn_masks'], pr)

    panels_with_strip = np.concatenate([panels, strip], axis=1)
    total_w = panels_with_strip.shape[1]

    fp_bar  = np.zeros((18, total_w, 3), dtype=np.uint8)
    fp_pil  = Image.fromarray(fp_bar)
    fp_draw = ImageDraw.Draw(fp_pil)
    summary = (f"FP-bg:{analysis['fp_bg_count']}  FP-crop:{analysis['fp_crop_count']}  "
               f"FN:{len(analysis['fn_masks'])}  "
               f"IoP>={iop_thr*100:.0f}%  IoG>={iog_thr*100:.0f}%")
    fp_draw.text((4, 3), summary, fill=(255, 255, 200))

    right_block = np.concatenate(
        [panels_with_strip, np.asarray(fp_pil, dtype=np.uint8)], axis=0)
    return np.concatenate([_make_legend_left(right_block.shape[0]), right_block], axis=1)


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


def _draw_pred_labels(draw: ImageDraw.Draw, analysis: dict,
                      cy1: int, cx1: int, ch: int, cw: int,
                      y_offset: int = 0) -> None:
    """Draw #N / FN#N labels on the prediction panel (3rd column) of a row PIL."""
    W = cw * 3
    for i, rec in enumerate(analysis['components'], 1):
        if not (rec['is_tp'] or rec['is_fp']):
            continue
        ys, xs = np.where(rec['mask'])
        if ys.size == 0:
            continue
        ry_min = int(ys.min()) - cy1
        ry_max = int(ys.max()) - cy1
        if ry_max < 0 or ry_min >= ch:
            continue
        ty = y_offset + max(14, ry_min - 10)
        tx = min(int(xs.min()) - cx1 + cw * 2, W - 25)
        _tight_text(draw, tx, ty, f'#{i}', (255, 255, 255), W)
    teal = tuple(_PALETTE['fn_weed'].tolist())
    for i, fn in enumerate(analysis['fn_masks'], 1):
        ys, xs = np.where(fn['mask'])
        if ys.size == 0:
            continue
        ry_min = int(ys.min()) - cy1
        ry_max = int(ys.max()) - cy1
        if ry_max < 0 or ry_min >= ch:
            continue
        ty = y_offset + max(14, ry_min - 10)
        tx = min(int(xs.min()) - cx1 + cw * 2, W - 30)
        _tight_text(draw, tx, ty, f'FN#{i}', teal, W)


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
    """2-row × 3-col crop comparison image with per-row info sidebar."""
    pr, pg, pp1 = render_full_panels(rgb, gt_np, pred1_np, analysis1, c0, c1, c2, ignore_index)
    _,   _,  pp2 = render_full_panels(rgb, gt_np, pred2_np, analysis2, c0, c1, c2, ignore_index)
    cw, ch = cx2 - cx1, cy2 - cy1
    W = cw * 3

    def _build_row(pp, analysis, hdr_text, hdr_color):
        arr = np.concatenate([_c(pr, cy1, cx1, cy2, cx2),
                              _c(pg, cy1, cx1, cy2, cx2),
                              _c(pp, cy1, cx1, cy2, cx2)], axis=1)
        pil  = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([(0, 0), (W - 1, 13)], fill=(0, 0, 0))
        draw.text((4, 2),        'RGB',     fill=(180, 180, 180))
        draw.text((cw + 4, 2),  'GT mask', fill=(180, 180, 180))
        draw.text((cw*2 + 4, 2), hdr_text,  fill=hdr_color)
        for col in (cw, cw * 2):
            draw.line([(col, 0), (col, ch - 1)], fill=(110, 110, 110), width=1)
        _draw_pred_labels(draw, analysis, cy1, cx1, ch, cw)
        return np.asarray(pil, dtype=np.uint8)

    row1_panels = _build_row(pp1, analysis1, f'▲ {name1}', (255, 220, 80))
    row2_panels = _build_row(pp2, analysis2, f'▼ {name2}', (80,  200, 255))

    strip1 = _make_info_strip(ch, analysis1['components'], analysis1['fn_masks'],
                               pr, model_name=f'▲ {name1}',
                               cy1=cy1, cx1=cx1, cy2=cy2, cx2=cx2)
    strip2 = _make_info_strip(ch, analysis2['components'], analysis2['fn_masks'],
                               pr, model_name=f'▼ {name2}',
                               cy1=cy1, cx1=cx1, cy2=cy2, cx2=cx2)

    full_row1 = np.concatenate([row1_panels, strip1], axis=1)
    full_row2 = np.concatenate([row2_panels, strip2], axis=1)
    divider   = np.full((2, full_row1.shape[1], 3), 160, dtype=np.uint8)

    m1 = (f'M1 fpbg={analysis1["fp_bg_count"]} fpcrop={analysis1["fp_crop_count"]}'
          f' fn={len(analysis1["fn_masks"])}')
    m2 = (f'M2 fpbg={analysis2["fp_bg_count"]} fpcrop={analysis2["fp_crop_count"]}'
          f' fn={len(analysis2["fn_masks"])}')
    footer_txt = (f'src:{fp_inst["source"]}  cat:{fp_inst["category"]}'
                  f'  a={fp_inst["area"]}  |  {m1}  |  {m2}'
                  f'  |  IoP>={iop_thr*100:.0f}% IoG>={iog_thr*100:.0f}%')
    fp_bar  = np.zeros((18, full_row1.shape[1], 3), dtype=np.uint8)
    fp_pil  = Image.fromarray(fp_bar)
    ImageDraw.Draw(fp_pil).text((4, 3), footer_txt, fill=(255, 255, 200))

    right_block = np.concatenate(
        [full_row1, divider, full_row2, np.asarray(fp_pil, dtype=np.uint8)], axis=0)
    return np.concatenate([_make_legend_left(right_block.shape[0]), right_block], axis=1)


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
