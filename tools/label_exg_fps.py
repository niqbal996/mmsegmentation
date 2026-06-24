#!/usr/bin/env python3
"""Interactive ExG labelling tool for tiny weed FP-on-background instances.

Workflow
--------
1. Sweep the validation set with the provided model.
2. Collect FP-on-background predicted components with area <= --max-area px.
3. Stratify by date-prefix (05-15, 05-26, 06-05) and draw --n-samples total.
4. For each instance show a 4-panel figure:
      RGB crop  |  Prediction overlay  |  ExG heatmap (FP)  |  ExG heatmap (background control)
   The control panel uses the same mask shape placed on a pure-background region
   of the same image, giving a grounded ExG baseline for comparison.
   The title shows FP ExG, control ExG, and their delta (FP − ctrl).
   Press Y/N buttons or keys to label.
5. Write labels to  <output-dir>/exg_labels.csv  (incremental — safe to quit).
6. Print per-field accuracy and an optimal ExG threshold.

Usage
-----
python tools/label_exg_fps.py \\
    configs/wacv/your_config.py \\
    /path/to/checkpoint.pth \\
    --output-dir ./tmp/exg_labels \\
    --n-samples 50 --max-area 200 --device cuda:0
"""

import argparse
import csv
import json
import os
import os.path as osp
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as sp_ndimage

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.apis import init_model
from mmseg.registry import DATASETS

# ── Field date prefixes ────────────────────────────────────────────────────────
DATE_PREFIXES = ('05-15', '05-26', '06-05')

# ── Palette ────────────────────────────────────────────────────────────────────
_P_BG   = np.array([40,  40,  40],  dtype=np.uint8)
_P_CROP = np.array([0,   200, 0],   dtype=np.uint8)
_P_WEED = np.array([200, 0,   0],   dtype=np.uint8)
_P_FP   = np.array([255, 230, 0],   dtype=np.uint8)
_P_WHT  = np.array([255, 255, 255], dtype=np.uint8)


# ── Low-level image helpers ────────────────────────────────────────────────────

def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else float('nan')


def compute_exg(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Mean ExG = 2G - R - B over mask pixels."""
    if not mask.any():
        return float('nan')
    r = rgb[mask, 0].astype(np.float32)
    g = rgb[mask, 1].astype(np.float32)
    b = rgb[mask, 2].astype(np.float32)
    return float((2.0 * g - r - b).mean())


def exg_pixelmap(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel ExG = 2G - R - B, float32."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    return 2.0 * g - r - b


def compute_fixed_crop_bbox(cy: int, cx: int, h: int, w: int,
                             crop_h: int, crop_w: int) -> Tuple[int, int, int, int]:
    """Fixed crop_h x crop_w window centred on (cy, cx), clamped to image bounds."""
    y1, y2 = cy - crop_h // 2, cy - crop_h // 2 + crop_h
    x1, x2 = cx - crop_w // 2, cx - crop_w // 2 + crop_w
    if y1 < 0:   y2 -= y1;      y1 = 0
    if x1 < 0:   x2 -= x1;      x1 = 0
    if y2 > h:   y1 -= y2 - h;  y2 = h
    if x2 > w:   x1 -= x2 - w;  x2 = w
    return max(0, y1), max(0, x1), min(h, y2), min(w, x2)


def draw_contour(canvas: np.ndarray, mask: np.ndarray,
                 color: np.ndarray, thickness: int = 2) -> np.ndarray:
    if not mask.any():
        return canvas
    eroded = sp_ndimage.binary_erosion(
        mask, structure=np.ones((3, 3), dtype=bool), iterations=thickness)
    out = canvas.copy()
    out[mask & ~eroded] = color
    return out


def fill_overlay(canvas: np.ndarray, mask: np.ndarray,
                 color: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = canvas.astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + color.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Control region search ─────────────────────────────────────────────────────

# Candidate offsets: 8 directions × 3 distance multipliers (relative to crop_size)
_CTRL_DIRECTION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
_CTRL_DISTANCE_FACTORS = [1.2, 1.8, 2.5]


def find_control_crop(rgb: np.ndarray, gt_np: np.ndarray, pred_np: np.ndarray,
                      mask_crop: np.ndarray, cy: int, cx: int,
                      crop_size: int, c0: int, c2: int
                      ) -> Optional[Tuple[np.ndarray, float]]:
    """Find a background-only crop and compute ExG using the same mask shape.

    Tries candidate positions in multiple directions from the FP centroid.
    For each candidate the mask_crop shape is applied to the corresponding
    region in gt_np; positions where >= 90% of those pixels are background
    and no weed is predicted are eligible.  The candidate with the highest
    background ratio is returned as (rgb_control_crop, control_exg), or None
    if nothing suitable is found.
    """
    h, w = gt_np.shape
    mask_bool = mask_crop.astype(bool)
    if not mask_bool.any():
        return None

    best_rgb_crop: Optional[np.ndarray] = None
    best_exg: float = float('nan')
    best_score: float = -1.0

    for angle_deg in _CTRL_DIRECTION_ANGLES:
        rad = np.deg2rad(angle_deg)
        dy_unit = np.sin(rad)
        dx_unit = np.cos(rad)
        for factor in _CTRL_DISTANCE_FACTORS:
            new_cy = int(round(cy + factor * crop_size * dy_unit))
            new_cx = int(round(cx + factor * crop_size * dx_unit))

            # Clamp centre so the crop window stays inside the image
            new_cy = int(np.clip(new_cy, crop_size // 2, h - crop_size // 2))
            new_cx = int(np.clip(new_cx, crop_size // 2, w - crop_size // 2))

            ky1, kx1, ky2, kx2 = compute_fixed_crop_bbox(
                new_cy, new_cx, h, w, crop_size, crop_size)

            ctrl_gt   = gt_np[ky1:ky2, kx1:kx2]
            ctrl_pred = pred_np[ky1:ky2, kx1:kx2]

            if ctrl_gt.shape != mask_bool.shape:
                continue  # crop landed on image edge with different size

            bg_ratio    = float((ctrl_gt[mask_bool]   == c0).mean())
            noweed_ratio = float((ctrl_pred[mask_bool] != c2).mean())

            # Require the control region to be dominantly background
            if bg_ratio < 0.90:
                continue

            score = bg_ratio + 0.3 * noweed_ratio
            if score > best_score:
                ctrl_rgb = rgb[ky1:ky2, kx1:kx2]
                ctrl_exg = compute_exg(ctrl_rgb, mask_bool)
                best_score    = score
                best_rgb_crop = ctrl_rgb.copy()
                best_exg      = ctrl_exg

    if best_rgb_crop is None:
        return None
    return best_rgb_crop, best_exg


# ── FP-bg analysis ─────────────────────────────────────────────────────────────

def analyse_fp_bg(pred_np: np.ndarray, gt_np: np.ndarray,
                  c0: int, c1: int, c2: int,
                  ignore_index: int, iop_thr: float,
                  min_area: int, max_area: int) -> List[Dict]:
    """Return tiny FP-on-background weed component records."""
    pred_mask = pred_np == c2
    pred_labeled, n_pred = sp_ndimage.label(pred_mask)
    gt_weed = gt_np == c2

    records = []
    for cid in range(1, n_pred + 1):
        cm = pred_labeled == cid
        area = int(cm.sum())
        if area < min_area or area > max_area:
            continue

        iop = _safe_div(int((cm & gt_weed).sum()), area)
        if iop >= iop_thr:
            continue  # counts as TP — skip

        gt_under = gt_np[cm]
        dom = int(gt_under[np.argmax(np.bincount(gt_under))]) if gt_under.size > 0 else -1
        if dom != c0:
            continue  # not dominantly on background

        ys, xs = np.where(cm)
        cy = int(round(ys.mean()))
        cx = int(round(xs.mean()))
        records.append(dict(comp_id=cid, area=area, mask=cm, centroid=(cy, cx)))
    return records


# ── Dataset / model helpers ────────────────────────────────────────────────────

def build_val_dataset(cfg: Config):
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    val_dl = cfg.get('val_dataloader', None)
    if val_dl is None:
        raise RuntimeError('Config has no val_dataloader.')
    ds = DATASETS.build(val_dl['dataset'])
    ds.full_init()
    return ds


def run_model(model, data: dict, gt_np: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        results = model.test_step(
            {'inputs': [data['inputs']], 'data_samples': [data['data_samples']]})
    pred = results[0].pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int64)
    if pred.shape != gt_np.shape:
        pred = F.interpolate(
            torch.from_numpy(pred)[None, None].float(),
            size=gt_np.shape, mode='nearest')[0, 0].numpy().astype(np.int64)
    return pred


def _date_prefix(img_path: str) -> str:
    """Extract date prefix (e.g. '05-15') from image filename."""
    stem = osp.splitext(osp.basename(img_path))[0]
    for p in DATE_PREFIXES:
        if p in stem:
            return p
    m = re.search(r'(\d{2}-\d{2})', stem)
    return m.group(1) if m else 'unknown'


# ── Instance collection ────────────────────────────────────────────────────────

def collect_instances(model, dataset, c0: int, c1: int, c2: int,
                      ignore_index: int, iop_thr: float,
                      min_area: int, max_area: int, crop_size: int,
                      max_per_prefix: int = 200) -> List[Dict]:
    """Sweep the validation set and collect pre-cropped FP-bg instances.

    Collection is capped **per date-prefix** (max_per_prefix instances each),
    not globally.  Images whose prefix already has enough instances are skipped
    without running the model, so the scan continues until all prefixes have
    their quota or all images are exhausted.  This ensures balanced coverage
    across the three field dates even when images are sorted by filename.
    """
    collected: List[Dict] = []
    per_prefix: Dict[str, int] = defaultdict(int)
    print(f'Collecting FP-bg instances  area=[{min_area}, {max_area}] px'
          f'  max_per_prefix={max_per_prefix} ...')

    for idx in range(len(dataset)):
        # Early-exit only when ALL known prefixes are saturated
        if (len(per_prefix) == len(DATE_PREFIXES)
                and all(per_prefix[p] >= max_per_prefix for p in DATE_PREFIXES)):
            print('  All date prefixes saturated, stopping early.')
            break

        info = dataset.get_data_info(idx)
        img_path = info.get('img_path', '')
        seg_path = info.get('seg_map_path', '')
        if not img_path or not osp.exists(img_path):
            continue
        if not seg_path or not osp.exists(seg_path):
            continue

        # Resolve prefix BEFORE loading the model — cheap path to skip
        date_pfx = _date_prefix(img_path)
        if per_prefix[date_pfx] >= max_per_prefix:
            continue  # this prefix is full; skip without running the model

        try:
            data = dataset[idx]
        except Exception as e:
            print(f'  [idx={idx}] pipeline error: {e}')
            continue

        gt_np = (data['data_samples'].gt_sem_seg.data
                 .squeeze().cpu().numpy().astype(np.int64))
        h, w = gt_np.shape
        pred_np = run_model(model, data, gt_np)

        records = analyse_fp_bg(pred_np, gt_np, c0, c1, c2, ignore_index,
                                 iop_thr, min_area, max_area)
        if not records:
            continue

        rgb = np.asarray(
            Image.open(img_path).convert('RGB').resize((w, h), Image.BILINEAR),
            dtype=np.uint8)
        img_name = osp.basename(img_path)
        n_added  = 0

        for rec in records:
            if per_prefix[date_pfx] >= max_per_prefix:
                break  # hit cap mid-image; discard remaining components

            cy, cx = rec['centroid']
            cy1, cx1, cy2, cx2 = compute_fixed_crop_bbox(
                cy, cx, h, w, crop_size, crop_size)

            rgb_crop  = rgb[cy1:cy2, cx1:cx2].copy()
            pred_crop = pred_np[cy1:cy2, cx1:cx2].copy()
            mask_crop = rec['mask'][cy1:cy2, cx1:cx2].copy()
            exg_val   = compute_exg(rgb_crop, mask_crop)

            ctrl = find_control_crop(rgb, gt_np, pred_np, mask_crop,
                                     cy, cx, crop_size, c0, c2)
            ctrl_rgb_crop = ctrl[0] if ctrl is not None else None
            ctrl_exg      = ctrl[1] if ctrl is not None else float('nan')

            collected.append(dict(
                img_path=img_path,
                img_name=img_name,
                date_prefix=date_pfx,
                area=rec['area'],
                exg=exg_val,
                ctrl_exg=ctrl_exg,
                rgb_crop=rgb_crop,
                pred_crop=pred_crop,
                mask_crop=mask_crop,
                ctrl_rgb_crop=ctrl_rgb_crop,
            ))
            per_prefix[date_pfx] += 1
            n_added += 1

        if n_added:
            counts_str = '  '.join(f'{p}:{per_prefix[p]}' for p in DATE_PREFIXES)
            print(f'  [{idx + 1:4d}/{len(dataset)}] +{n_added} from {img_name}'
                  f'  |  {counts_str}')

    print(f'\nCollection complete: {len(collected)} instances total')
    for p in DATE_PREFIXES:
        print(f'  {p}: {per_prefix[p]}')
    return collected


# ── Stratified sampling ────────────────────────────────────────────────────────

def stratified_sample(instances: List[Dict], n_total: int) -> List[Dict]:
    """Sample n_total instances balanced across the three date prefixes."""
    by_prefix: Dict[str, List] = defaultdict(list)
    for inst in instances:
        by_prefix[inst['date_prefix']].append(inst)

    print('Availability by date prefix:')
    for p in sorted(by_prefix):
        print(f'  {p}: {len(by_prefix[p])} instances')

    active = [p for p in DATE_PREFIXES if p in by_prefix and by_prefix[p]]
    if not active:
        # No recognised prefixes — just shuffle and take
        pool = list(instances)
        random.shuffle(pool)
        return pool[:n_total]

    per_prefix = n_total // len(active)
    sampled: List[Dict] = []
    for p in active:
        pool = list(by_prefix[p])
        random.shuffle(pool)
        sampled.extend(pool[:per_prefix])

    # Top up if short (can happen when some prefix has too few instances)
    if len(sampled) < n_total:
        used = {(i['img_path'], i['area']) for i in sampled}
        extra = [i for i in instances if (i['img_path'], i['area']) not in used]
        random.shuffle(extra)
        sampled.extend(extra[:n_total - len(sampled)])

    random.shuffle(sampled)
    return sampled[:n_total]


# ── Panel rendering ────────────────────────────────────────────────────────────

def render_pred_panel(rgb_crop: np.ndarray, pred_crop: np.ndarray,
                      mask_crop: np.ndarray,
                      c0: int, c1: int, c2: int) -> np.ndarray:
    h, w = pred_crop.shape
    panel = np.full((h, w, 3), _P_BG, dtype=np.uint8)
    panel[pred_crop == c1] = _P_CROP
    panel[pred_crop == c2] = _P_WEED
    panel = fill_overlay(panel, mask_crop, _P_FP, alpha=0.6)
    panel = draw_contour(panel, mask_crop, _P_FP, thickness=2)
    return panel


def render_exg_panel(rgb_crop: np.ndarray, mask_crop: np.ndarray,
                     vmin: float = -50.0, vmax: float = 100.0) -> np.ndarray:
    """Composite ExG panel: dimmed RGB everywhere, ExG heatmap only on the mask.

    Non-mask pixels are shown at 40 % brightness so the labeller can see the
    surrounding scene for context.  Only the mask pixels receive the RdYlGn
    ExG colourmap, making it unambiguous which area the scalar ExG value
    refers to.  A white contour marks the mask boundary.
    """
    import matplotlib.cm as cm

    mask  = mask_crop.astype(bool)
    panel = rgb_crop.astype(np.float32).copy()

    # Dim everything outside the mask
    panel[~mask] *= 0.40

    # Paint ExG heatmap on mask pixels only
    if mask.any():
        exg    = exg_pixelmap(rgb_crop)
        normed = np.clip((exg - vmin) / (vmax - vmin), 0.0, 1.0)
        exg_rgb = (cm.RdYlGn(normed)[:, :, :3] * 255).astype(np.float32)
        panel[mask] = exg_rgb[mask]

    panel = np.clip(panel, 0, 255).astype(np.uint8)
    panel = draw_contour(panel, mask_crop, _P_WHT, thickness=2)
    return panel


# ── Interactive labelling loop ─────────────────────────────────────────────────

def label_instances_interactive(instances: List[Dict],
                                 c0: int, c1: int, c2: int,
                                 exg_threshold: float,
                                 output_dir: str) -> Tuple[List[Dict], str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button as MplButton
    except ImportError:
        print('ERROR: matplotlib is required for interactive labelling.')
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = osp.join(output_dir, 'exg_labels.csv')
    fieldnames = ['index', 'img_name', 'date_prefix', 'area',
                  'exg', 'ctrl_exg', 'exg_delta', 'is_plant', 'img_path']
    csv_f = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_w = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction='ignore')
    csv_w.writeheader()

    labels: List[Dict] = []
    total = len(instances)
    _state: Dict = {'choice': None}

    def _on_key(event):
        if event.key in ('y', 'Y'):
            _state['choice'] = True
            plt.close('all')
        elif event.key in ('n', 'N'):
            _state['choice'] = False
            plt.close('all')
        elif event.key in ('q', 'Q', 'escape'):
            _state['choice'] = 'quit'
            plt.close('all')

    print(f'\nStarting labelling: {total} instances.')
    print('Controls:  Y = yes (plant)  |  N = no (not a plant)  |  Q = quit & save')
    print()

    for i, inst in enumerate(instances):
        exg_val  = inst['exg']
        ctrl_exg = inst.get('ctrl_exg', float('nan'))
        has_ctrl = inst.get('ctrl_rgb_crop') is not None and not np.isnan(ctrl_exg)

        exg_str  = f'{exg_val:+.1f}' if not np.isnan(exg_val) else 'NaN'
        ctrl_str = f'{ctrl_exg:+.1f}' if has_ctrl else 'N/A'
        delta    = (exg_val - ctrl_exg
                    if (not np.isnan(exg_val) and has_ctrl) else float('nan'))
        delta_str = f'{delta:+.1f}' if not np.isnan(delta) else 'N/A'
        exg_hint  = ('HIGH' if (not np.isnan(exg_val) and exg_val >= exg_threshold)
                     else 'low')

        # ── Build panels ────────────────────────────────────────────────────
        panel_rgb  = inst['rgb_crop']
        panel_pred = render_pred_panel(inst['rgb_crop'], inst['pred_crop'],
                                       inst['mask_crop'], c0, c1, c2)
        panel_exg  = render_exg_panel(inst['rgb_crop'], inst['mask_crop'])

        n_cols = 4 if has_ctrl else 3
        fig_w  = 17.0 if has_ctrl else 13.0

        fig = plt.figure(figsize=(fig_w, 5.5), facecolor='#111111')

        title_line1 = (
            f'[{i + 1}/{total}]  {inst["img_name"]}  |  field: {inst["date_prefix"]}'
            f'  |  area: {inst["area"]} px')
        title_line2 = (
            f'FP ExG = {exg_str} ({exg_hint})   '
            + (f'Ctrl ExG = {ctrl_str}   Delta = {delta_str}   '
               if has_ctrl else '')
            + f'threshold = {exg_threshold:+.0f}')
        title_line3 = 'Y = plant (yes)    N = not a plant (no)    Q = quit & save'
        fig.suptitle(f'{title_line1}\n{title_line2}\n{title_line3}',
                     color='white', fontsize=9, y=0.99)

        gs = fig.add_gridspec(1, n_cols, left=0.03, right=0.97,
                               bottom=0.18, top=0.86, wspace=0.05)
        axes = [fig.add_subplot(gs[0, j]) for j in range(n_cols)]

        col_panels = [panel_rgb, panel_pred, panel_exg]
        col_titles = [
            'RGB (FP region)',
            'Prediction  (FP-bg = yellow)',
            f'ExG heatmap — FP  ({exg_str})',
        ]
        if has_ctrl:
            panel_ctrl = render_exg_panel(inst['ctrl_rgb_crop'], inst['mask_crop'])
            col_panels.append(panel_ctrl)
            col_titles.append(f'ExG heatmap — BG control  ({ctrl_str})')

        for ax, panel, title in zip(axes, col_panels, col_titles):
            ax.imshow(panel)
            ax.set_title(title, color='white', fontsize=8, pad=3)
            ax.axis('off')

        # Annotate ExG value on FP panel
        _exg_bbox = dict(boxstyle='round,pad=0.3', fc='#222222', alpha=0.85)
        axes[2].text(0.03, 0.05, f'FP  ExG = {exg_str}',
                     transform=axes[2].transAxes, color='#aaffaa',
                     fontsize=8, bbox=_exg_bbox)

        # Annotate control ExG and delta on control panel
        if has_ctrl:
            delta_color = '#aaffaa' if delta > 0 else '#ffaaaa'
            axes[3].text(
                0.03, 0.05,
                f'Ctrl ExG = {ctrl_str}\nDelta = {delta_str}',
                transform=axes[3].transAxes, color=delta_color,
                fontsize=8, bbox=_exg_bbox)

        # ── Buttons ─────────────────────────────────────────────────────────
        ax_yes = fig.add_axes([0.30, 0.04, 0.17, 0.09])
        ax_no  = fig.add_axes([0.53, 0.04, 0.17, 0.09])
        btn_yes = MplButton(ax_yes, 'Yes (Y)  =  plant',
                            color='#1a4d1a', hovercolor='#2d7a2d')
        btn_no  = MplButton(ax_no,  'No (N)  =  not plant',
                            color='#4d1a1a', hovercolor='#7a2d2d')
        btn_yes.label.set_color('white')
        btn_no.label.set_color('white')

        def _yes(ev, s=_state): s['choice'] = True;  plt.close('all')
        def _no(ev,  s=_state): s['choice'] = False; plt.close('all')
        btn_yes.on_clicked(_yes)
        btn_no.on_clicked(_no)
        fig.canvas.mpl_connect('key_press_event', _on_key)

        _state['choice'] = None
        try:
            plt.show(block=True)
        except Exception as e:
            print(f'\nDisplay error: {e}')
            print('Tip: on WSL2 ensure WSLg is running, or set DISPLAY for X11.')
            csv_f.close()
            sys.exit(1)

        choice = _state['choice']
        if choice == 'quit':
            print(f'Quit after {i} labelled instances.')
            break

        is_plant = bool(choice) if choice is not None else False
        record = dict(
            index=i + 1,
            img_path=inst['img_path'],
            img_name=inst['img_name'],
            date_prefix=inst['date_prefix'],
            area=inst['area'],
            exg=float(exg_val)  if not np.isnan(exg_val)  else None,
            ctrl_exg=float(ctrl_exg) if has_ctrl          else None,
            exg_delta=float(delta) if not np.isnan(delta) else None,
            is_plant=is_plant,
        )
        labels.append(record)
        csv_w.writerow(record)
        csv_f.flush()

        ctrl_note = f'  ctrl={ctrl_str}  Δ={delta_str}' if has_ctrl else ''
        verdict   = 'PLANT' if is_plant else 'not-plant'
        print(f'  [{i + 1:3d}/{total}]  {inst["img_name"]}'
              f'  ExG={exg_str}{ctrl_note}  area={inst["area"]} => {verdict}')

    csv_f.close()
    return labels, csv_path


# ── Statistics + threshold analysis ───────────────────────────────────────────

def compute_stats(labels: List[Dict]) -> None:
    if not labels:
        print('No labels recorded.')
        return

    print('\n' + '=' * 68)
    print('LABELLING RESULTS')
    print('=' * 68)

    total = len(labels)
    n_plant = sum(1 for l in labels if l['is_plant'])
    n_no    = total - n_plant
    print(f'Total labelled   : {total}')
    print(f'  Plant  (yes)   : {n_plant}  ({100 * n_plant / total:.1f}%)')
    print(f'  Non-plant (no) : {n_no}   ({100 * n_no / total:.1f}%)')

    by_prefix: Dict[str, List] = defaultdict(list)
    for l in labels:
        by_prefix[l['date_prefix']].append(l)

    print('\nPer-field breakdown:')
    for p in sorted(by_prefix):
        grp = by_prefix[p]
        np_ = sum(1 for l in grp if l['is_plant'])
        print(f'  {p}: {np_}/{len(grp)} plants  ({100 * np_ / len(grp):.1f}%)')

    valid = [l for l in labels if l['exg'] is not None]
    if not valid:
        return

    exg_arr   = np.array([l['exg'] for l in valid])
    plant_arr = np.array([l['is_plant'] for l in valid])

    if plant_arr.any():
        print(f'\nExG stats — Plants    :'
              f'  mean={exg_arr[plant_arr].mean():+.1f}'
              f'  std={exg_arr[plant_arr].std():.1f}'
              f'  min={exg_arr[plant_arr].min():+.1f}'
              f'  max={exg_arr[plant_arr].max():+.1f}')
    if (~plant_arr).any():
        print(f'ExG stats — Non-plant :'
              f'  mean={exg_arr[~plant_arr].mean():+.1f}'
              f'  std={exg_arr[~plant_arr].std():.1f}'
              f'  min={exg_arr[~plant_arr].min():+.1f}'
              f'  max={exg_arr[~plant_arr].max():+.1f}')

    def _thr_search(values: np.ndarray, labels_bool: np.ndarray,
                    label: str, default_best: float = 15.0) -> None:
        """Print threshold sweep table for one feature."""
        thresholds  = np.arange(float(values.min()) - 1,
                                float(values.max()) + 1, 0.5)
        best_acc, best_thr = 0.0, default_best
        rows = []
        for t in thresholds:
            pp = values >= t
            tp = int((pp & labels_bool).sum())
            tn = int((~pp & ~labels_bool).sum())
            fp = int((pp & ~labels_bool).sum())
            fn = int((~pp & labels_bool).sum())
            acc = (tp + tn) / len(values)
            rows.append((t, acc, tp, tn, fp, fn))
            if acc > best_acc:
                best_acc, best_thr = acc, float(t)
        print(f'\n{label} threshold search:')
        print(f'  Optimal : >= {best_thr:+.1f}   accuracy = {best_acc * 100:.1f}%')
        print(f'  {"Threshold":>10}  {"Acc":>7}  {"TP":>5}  {"TN":>5}'
              f'  {"FP":>5}  {"FN":>5}')
        for t, acc, tp, tn, fp, fn in rows:
            if abs(t - best_thr) > 12:
                continue
            marker = ' <-- best' if t == best_thr else ''
            print(f'  {t:>+10.1f}  {acc * 100:>6.1f}%  {tp:>5}  {tn:>5}'
                  f'  {fp:>5}  {fn:>5}{marker}')

    _thr_search(exg_arr, plant_arr, 'Raw FP ExG')

    # Delta (FP ExG − control ExG) threshold search — only when control data exists
    delta_valid = [l for l in labels
                   if l.get('exg_delta') is not None and l['exg'] is not None]
    if delta_valid:
        delta_arr  = np.array([l['exg_delta'] for l in delta_valid])
        plant_d    = np.array([l['is_plant']  for l in delta_valid])
        ctrl_arr   = np.array([l['ctrl_exg']  for l in delta_valid])

        if plant_d.any():
            print(f'\nDelta (FP−ctrl) stats — Plants    :'
                  f'  mean={delta_arr[plant_d].mean():+.1f}'
                  f'  std={delta_arr[plant_d].std():.1f}')
        if (~plant_d).any():
            print(f'Delta (FP−ctrl) stats — Non-plant :'
                  f'  mean={delta_arr[~plant_d].mean():+.1f}'
                  f'  std={delta_arr[~plant_d].std():.1f}')

        _thr_search(delta_arr, plant_d, 'ExG Delta (FP − background control)',
                    default_best=0.0)

        # Also print a quick summary of raw control ExG for reference
        print(f'\nControl (background) ExG across all labelled:'
              f'  mean={ctrl_arr.mean():+.1f}  std={ctrl_arr.std():.1f}'
              f'  min={ctrl_arr.min():+.1f}  max={ctrl_arr.max():+.1f}')

    # Per-field ExG breakdown
    print('\nPer-field ExG statistics:')
    for p in sorted(by_prefix):
        grp = by_prefix[p]
        exg_g = np.array([l['exg'] for l in grp if l['exg'] is not None])
        if exg_g.size == 0:
            continue
        pla_g  = np.array([l['exg'] for l in grp if l['is_plant'] and l['exg'] is not None])
        nop_g  = np.array([l['exg'] for l in grp if not l['is_plant'] and l['exg'] is not None])
        dlt_g  = np.array([l['exg_delta'] for l in grp
                           if l.get('exg_delta') is not None])
        print(f'  {p}:')
        print(f'    FP ExG   mean={exg_g.mean():+.1f}  std={exg_g.std():.1f}'
              f'  min={exg_g.min():+.1f}  max={exg_g.max():+.1f}')
        if dlt_g.size > 0:
            print(f'    Delta    mean={dlt_g.mean():+.1f}  std={dlt_g.std():.1f}'
                  f'  min={dlt_g.min():+.1f}  max={dlt_g.max():+.1f}')
        if pla_g.size > 0:
            print(f'    Plant    FP ExG mean={pla_g.mean():+.1f}  std={pla_g.std():.1f}')
        if nop_g.size > 0:
            print(f'    NonPlt   FP ExG mean={nop_g.mean():+.1f}  std={nop_g.std():.1f}')


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config',
                   help='MMSeg config file')
    p.add_argument('checkpoint',
                   help='Model checkpoint (.pth)')
    p.add_argument('--output-dir', required=True,
                   help='Directory for labels CSV / JSON')
    p.add_argument('--n-samples',    type=int,   default=60,
                   help='Target instances to label; 20 per field condition (default 60)')
    p.add_argument('--max-area',     type=int,   default=200,
                   help='Maximum FP-bg component area in pixels (default 200)')
    p.add_argument('--min-area',     type=int,   default=5,
                   help='Minimum FP-bg component area in pixels (default 5)')
    p.add_argument('--crop-size',    type=int,   default=256,
                   help='Fixed square crop window (px) centred on instance (default 256)')
    p.add_argument('--exg-threshold', type=float, default=15.0,
                   help='Initial ExG threshold for the "HIGH/low" hint (default 15)')
    p.add_argument('--iop-thr',      type=float, default=0.05,
                   help='IoP threshold to treat a prediction as TP (default 0.05)')
    p.add_argument('--device',       default='cuda:0')
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--class0-label', type=int,   default=0,
                   help='Label index for background class (default 0)')
    p.add_argument('--class1-label', type=int,   default=1,
                   help='Label index for crop class (default 1)')
    p.add_argument('--class2-label', type=int,   default=2,
                   help='Label index for weed class (default 2)')
    p.add_argument('--ignore-index', type=int,   default=255)
    p.add_argument('--max-per-prefix', type=int, default=200,
                   help='Max instances collected per date prefix (default 200). '
                        'Scanning continues until all three prefixes reach this '
                        'cap or all images are exhausted, so no date is starved '
                        'by images from another date appearing first.')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    c0, c1, c2 = args.class0_label, args.class1_label, args.class2_label

    print(f'Config       : {args.config}')
    print(f'Checkpoint   : {args.checkpoint}')
    print(f'Output dir   : {args.output_dir}')
    print(f'Labels       : bg={c0}  crop={c1}  weed={c2}  ignore={args.ignore_index}')
    print(f'Filter       : area [{args.min_area}, {args.max_area}] px  IoP < {args.iop_thr}')
    print(f'Target       : {args.n_samples} samples  crop {args.crop_size}x{args.crop_size} px')
    print(f'Max/prefix   : {args.max_per_prefix} instances per date prefix')
    print(f'ExG hint thr : {args.exg_threshold:+.1f}')
    print()

    cfg     = Config.fromfile(args.config)
    dataset = build_val_dataset(cfg)
    print(f'Validation set: {len(dataset)} images')

    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()

    instances = collect_instances(
        model, dataset, c0, c1, c2, args.ignore_index,
        args.iop_thr, args.min_area, args.max_area,
        args.crop_size, args.max_per_prefix)

    if not instances:
        print('No FP-bg instances found. Try increasing --max-area or check checkpoint.')
        sys.exit(1)

    print('\nSampling ...')
    sampled = stratified_sample(instances, args.n_samples)
    print(f'Sampled {len(sampled)} instances for labelling.')

    labels, csv_path = label_instances_interactive(
        sampled, c0, c1, c2, args.exg_threshold, args.output_dir)

    if not labels:
        print('No labels recorded.')
        return

    json_path = osp.join(args.output_dir, 'exg_labels.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)

    print(f'\nLabels saved to:')
    print(f'  CSV  : {csv_path}')
    print(f'  JSON : {json_path}')

    compute_stats(labels)


if __name__ == '__main__':
    main()
