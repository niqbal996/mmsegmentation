#!/usr/bin/env python3
"""Analyse ExG labelling results from multiple labellers.

Given CSV files from 2–4 labellers (output of tools/label_exg_fps.py):
  1. Matches instances across labellers on (img_name, area, rounded-ExG).
  2. Classifies each instance as agreed-plant / agreed-not-plant / disputed.
  3. Computes per-labeller and overall ExG statistics.
  4. Finds the optimal ExG threshold that maximises accuracy on agreed instances.
  5. Saves a figure showing ExG distributions with mean lines and threshold.

Usage
-----
python tools/analyze_exg_labels.py \\
    labeller1/exg_labels.csv  labeller2/exg_labels.csv  labeller3/exg_labels.csv \\
    --output ./tmp/exg_analysis \\
    --names Alice Bob Carol
"""

import argparse
import csv
import json
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import gaussian_kde


# ── Loading ────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> List[Dict]:
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    records = []
    for r in rows:
        try:
            exg = float(r['exg']) if r.get('exg') not in ('', 'None', None) else None
            ctrl_exg = (float(r['ctrl_exg'])
                        if r.get('ctrl_exg') not in ('', 'None', None) else None)
            exg_delta = (float(r['exg_delta'])
                         if r.get('exg_delta') not in ('', 'None', None) else None)
            records.append(dict(
                img_name=r['img_name'],
                date_prefix=r.get('date_prefix', 'unknown'),
                area=int(r['area']),
                exg=exg,
                ctrl_exg=ctrl_exg,
                exg_delta=exg_delta,
                is_plant=r['is_plant'].strip().lower() in ('true', '1', 'yes'),
            ))
        except (KeyError, ValueError):
            continue
    return records


def load_json(path: str) -> List[Dict]:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for r in data:
        exg = r.get('exg')
        if isinstance(exg, str):
            exg = float(exg) if exg not in ('', 'None') else None
        ctrl_exg  = r.get('ctrl_exg')
        exg_delta = r.get('exg_delta')
        records.append(dict(
            img_name=r['img_name'],
            date_prefix=r.get('date_prefix', 'unknown'),
            area=int(r['area']),
            exg=exg,
            ctrl_exg=float(ctrl_exg)  if ctrl_exg  is not None else None,
            exg_delta=float(exg_delta) if exg_delta is not None else None,
            is_plant=bool(r['is_plant']),
        ))
    return records


def load_file(path: str) -> List[Dict]:
    if path.endswith('.json'):
        return load_json(path)
    return load_csv(path)


# ── Instance matching ──────────────────────────────────────────────────────────

def _instance_key(rec: Dict) -> Tuple:
    """Stable key for matching the same FP component across labeller files."""
    exg = rec['exg']
    exg_rounded = round(exg, 1) if exg is not None else None
    return (rec['img_name'], rec['area'], exg_rounded)


def merge_labellers(labeller_records: List[List[Dict]],
                    names: List[str]) -> Dict[Tuple, Dict]:
    """Return a dict keyed by instance key.

    Each value is:
        {
          'votes':       {labeller_name: is_plant},
          'exg':         float | None,
          'ctrl_exg':    float | None,
          'exg_delta':   float | None,
          'date_prefix': str,
          'area':        int,
        }
    """
    merged: Dict[Tuple, Dict] = {}
    for labeller_recs, name in zip(labeller_records, names):
        for rec in labeller_recs:
            key = _instance_key(rec)
            if key not in merged:
                merged[key] = dict(
                    votes={},
                    exg=rec['exg'],
                    ctrl_exg=rec.get('ctrl_exg'),
                    exg_delta=rec.get('exg_delta'),
                    date_prefix=rec['date_prefix'],
                    area=rec['area'],
                    img_name=rec['img_name'],
                )
            merged[key]['votes'][name] = rec['is_plant']
    return merged


def classify_instances(merged: Dict[Tuple, Dict],
                        majority: bool = True
                        ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split merged instances into agreed-plant, agreed-not-plant, disputed.

    If majority=True, use majority vote (>=50% of labellers say plant).
    If majority=False, require unanimity for agreement.
    """
    plants, nonplants, disputed = [], [], []
    for key, inst in merged.items():
        votes = list(inst['votes'].values())
        n_yes = sum(votes)
        n_total = len(votes)
        if majority:
            is_agreed_plant    = n_yes > n_total / 2
            is_agreed_nonplant = n_yes <= n_total / 2 and n_yes == 0
            # For majority: agreed-nonplant = all said no; disputed = split
            # Actually: plant if majority yes, nonplant if majority no, disputed if tied
            if n_yes > n_total / 2:
                plants.append(inst)
            elif n_yes < n_total / 2:
                nonplants.append(inst)
            else:
                disputed.append(inst)
        else:
            # Unanimity
            if all(votes):
                plants.append(inst)
            elif not any(votes):
                nonplants.append(inst)
            else:
                disputed.append(inst)
    return plants, nonplants, disputed


# ── Threshold analysis ─────────────────────────────────────────────────────────

def threshold_sweep(plant_exg: np.ndarray, nonplant_exg: np.ndarray,
                    step: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Sweep ExG thresholds and compute accuracy at each.

    Returns (thresholds, accuracies, best_threshold, best_accuracy).
    """
    all_exg = np.concatenate([plant_exg, nonplant_exg])
    thresholds = np.arange(all_exg.min() - 2, all_exg.max() + 2, step)
    n = len(plant_exg) + len(nonplant_exg)
    accuracies = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        tp = int((plant_exg    >= t).sum())
        tn = int((nonplant_exg <  t).sum())
        accuracies[i] = (tp + tn) / n

    best_i   = int(np.argmax(accuracies))
    best_thr = float(thresholds[best_i])
    best_acc = float(accuracies[best_i])
    return thresholds, accuracies, best_thr, best_acc


# ── Plotting ───────────────────────────────────────────────────────────────────

def _kde(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(x_grid)
    bw = max(1.06 * values.std() * len(values) ** (-0.2), 1.0)
    return gaussian_kde(values, bw_method=bw / values.std())(x_grid)


def plot_analysis(plant_exg: np.ndarray, nonplant_exg: np.ndarray,
                  disputed_exg: np.ndarray,
                  labeller_data: List[Tuple[str, np.ndarray, np.ndarray]],
                  thresholds: np.ndarray, accuracies: np.ndarray,
                  best_thr: float, best_acc: float,
                  midpoint_thr: float,
                  output_path: str) -> None:
    """Two-panel figure: ExG distributions (left) + accuracy curve (right)."""

    fig = plt.figure(figsize=(14, 6), facecolor='white')
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.32,
                             left=0.07, right=0.97, top=0.88, bottom=0.12)
    ax_dist = fig.add_subplot(gs[0])
    ax_acc  = fig.add_subplot(gs[1])

    # ── Distribution panel ─────────────────────────────────────────────────────
    x_min = min(np.concatenate([plant_exg, nonplant_exg]).min() - 10, -30)
    x_max = max(np.concatenate([plant_exg, nonplant_exg]).max() + 10,  80)
    x_grid = np.linspace(x_min, x_max, 400)

    kde_plant    = _kde(plant_exg,    x_grid)
    kde_nonplant = _kde(nonplant_exg, x_grid)

    # Normalise so both curves share the same peak height
    peak = max(kde_plant.max(), kde_nonplant.max(), 1e-9)
    kde_plant    /= peak
    kde_nonplant /= peak

    ax_dist.fill_between(x_grid, kde_nonplant, alpha=0.20, color='#cc3333', label=None)
    ax_dist.fill_between(x_grid, kde_plant,    alpha=0.20, color='#2ca02c', label=None)
    ax_dist.plot(x_grid, kde_nonplant, color='#cc3333', lw=2,
                 label=f'Not-plant  n={len(nonplant_exg)}')
    ax_dist.plot(x_grid, kde_plant,    color='#2ca02c', lw=2,
                 label=f'Plant      n={len(plant_exg)}')

    if len(disputed_exg) > 0:
        kde_dis = _kde(disputed_exg, x_grid)
        kde_dis /= peak
        ax_dist.plot(x_grid, kde_dis, color='#888888', lw=1.5, ls='--',
                     label=f'Disputed   n={len(disputed_exg)}')

    # Rug plots (small ticks at bottom)
    rug_y = -0.025
    for v in nonplant_exg:
        ax_dist.axvline(v, ymin=0, ymax=0.015, color='#cc3333', alpha=0.5, lw=0.8)
    for v in plant_exg:
        ax_dist.axvline(v, ymin=0.015, ymax=0.030, color='#2ca02c', alpha=0.5, lw=0.8)

    mean_plant    = float(plant_exg.mean())    if len(plant_exg)    > 0 else float('nan')
    mean_nonplant = float(nonplant_exg.mean()) if len(nonplant_exg) > 0 else float('nan')

    ax_dist.axvline(mean_plant,    color='#2ca02c', lw=2, ls='--',
                    label=f'Mean plant    = {mean_plant:+.1f}')
    ax_dist.axvline(mean_nonplant, color='#cc3333', lw=2, ls='--',
                    label=f'Mean non-plant = {mean_nonplant:+.1f}')
    ax_dist.axvline(best_thr, color='black', lw=2.5, ls='-',
                    label=f'Optimal thr = {best_thr:+.1f}  (acc {best_acc*100:.0f}%)')
    if abs(midpoint_thr - best_thr) > 0.5:
        ax_dist.axvline(midpoint_thr, color='navy', lw=1.5, ls=':',
                        label=f'Midpoint thr = {midpoint_thr:+.1f}')

    # Per-labeller means as small markers on the x-axis
    LABELLER_COLORS = ['#e6771e', '#9467bd', '#17becf', '#8c564b']
    for j, (lname, l_plant, l_nonp) in enumerate(labeller_data):
        c = LABELLER_COLORS[j % len(LABELLER_COLORS)]
        if len(l_plant) > 0:
            ax_dist.scatter([l_plant.mean()], [-0.05], marker='^', color=c, s=60,
                            zorder=5, clip_on=False,
                            label=f'{lname}  plant mean = {l_plant.mean():+.1f}')
        if len(l_nonp) > 0:
            ax_dist.scatter([l_nonp.mean()], [-0.08], marker='v', color=c, s=60,
                            zorder=5, clip_on=False,
                            label=f'{lname}  non-plant mean = {l_nonp.mean():+.1f}')

    ax_dist.set_xlim(x_min, x_max)
    ax_dist.set_ylim(-0.12, 1.15)
    ax_dist.set_xlabel('ExG  (2G − R − B)', fontsize=12)
    ax_dist.set_ylabel('Density (normalised)', fontsize=12)
    ax_dist.set_title('ExG distribution — agreed plant vs not-plant', fontsize=12)
    ax_dist.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
    ax_dist.axhline(0, color='grey', lw=0.5)
    ax_dist.grid(axis='x', alpha=0.25)

    # ── Accuracy-vs-threshold panel ────────────────────────────────────────────
    ax_acc.plot(thresholds, accuracies * 100, color='steelblue', lw=2)
    ax_acc.axvline(best_thr, color='black', lw=2, ls='-',
                   label=f'Optimal = {best_thr:+.1f}')
    if abs(midpoint_thr - best_thr) > 0.5:
        ax_acc.axvline(midpoint_thr, color='navy', lw=1.5, ls=':',
                       label=f'Midpoint = {midpoint_thr:+.1f}')
    ax_acc.scatter([best_thr], [best_acc * 100], color='black', s=60, zorder=5)
    ax_acc.set_xlabel('ExG threshold', fontsize=12)
    ax_acc.set_ylabel('Accuracy  (%)', fontsize=12)
    ax_acc.set_title('Threshold accuracy', fontsize=12)
    ax_acc.legend(fontsize=8)
    ax_acc.set_ylim(0, 105)
    ax_acc.grid(alpha=0.3)

    fig.suptitle(
        f'ExG threshold analysis  —  {len(plant_exg)} agreed-plant  '
        f'{len(nonplant_exg)} agreed-not-plant  {len(disputed_exg)} disputed',
        fontsize=11)

    os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved: {output_path}')
    plt.show()


# ── Statistics printer ─────────────────────────────────────────────────────────

def print_stats(plants: List[Dict], nonplants: List[Dict], disputed: List[Dict],
                labeller_data: List[Tuple[str, np.ndarray, np.ndarray]],
                best_thr: float, best_acc: float, midpoint_thr: float) -> None:

    plant_exg    = np.array([i['exg'] for i in plants    if i['exg'] is not None])
    nonplant_exg = np.array([i['exg'] for i in nonplants if i['exg'] is not None])

    n_total = len(plants) + len(nonplants) + len(disputed)
    print('\n' + '=' * 65)
    print('ExG THRESHOLD ANALYSIS')
    print('=' * 65)
    print(f'Total matched instances : {n_total}')
    print(f'  Agreed plant          : {len(plants)}')
    print(f'  Agreed not-plant      : {len(nonplants)}')
    print(f'  Disputed              : {len(disputed)}')

    if len(plant_exg):
        print(f'\nExG — agreed plant   : mean={plant_exg.mean():+.2f}'
              f'  std={plant_exg.std():.2f}'
              f'  min={plant_exg.min():+.1f}  max={plant_exg.max():+.1f}')
    if len(nonplant_exg):
        print(f'ExG — agreed not-plt : mean={nonplant_exg.mean():+.2f}'
              f'  std={nonplant_exg.std():.2f}'
              f'  min={nonplant_exg.min():+.1f}  max={nonplant_exg.max():+.1f}')

    print(f'\nRecommended thresholds:')
    print(f'  Midpoint  (mean_plant + mean_nonplant) / 2 = {midpoint_thr:+.2f}')
    print(f'  Optimal   (max accuracy on agreed set)     = {best_thr:+.2f}'
          f'  ({best_acc*100:.1f}% accuracy)')

    print('\nPer-labeller breakdown:')
    for name, l_plant, l_nonp in labeller_data:
        n_p, n_n = len(l_plant), len(l_nonp)
        mp  = f'{l_plant.mean():+.1f}' if n_p > 0 else 'N/A'
        mn  = f'{l_nonp.mean():+.1f}'  if n_n > 0 else 'N/A'
        print(f'  {name:<12}  plant n={n_p} mean={mp}'
              f'   not-plant n={n_n} mean={mn}')

    # Per-field breakdown
    by_prefix: Dict[str, Dict] = defaultdict(lambda: {'plant': [], 'nonplant': []})
    for inst in plants:
        by_prefix[inst['date_prefix']]['plant'].append(inst['exg'])
    for inst in nonplants:
        by_prefix[inst['date_prefix']]['nonplant'].append(inst['exg'])
    print('\nPer-field (agreed instances):')
    for p in sorted(by_prefix):
        pla = [v for v in by_prefix[p]['plant']    if v is not None]
        nop = [v for v in by_prefix[p]['nonplant'] if v is not None]
        mp  = f'{np.mean(pla):+.1f}' if pla else 'N/A'
        mn  = f'{np.mean(nop):+.1f}' if nop else 'N/A'
        print(f'  {p}  plant n={len(pla)} mean={mp}   not-plant n={len(nop)} mean={mn}')


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='+',
                   help='CSV or JSON label files, one per labeller')
    p.add_argument('--names', nargs='+', default=None,
                   help='Labeller names (default: Labeller1, Labeller2, ...)')
    p.add_argument('--output', default='./tmp/exg_analysis',
                   help='Output directory (default: ./tmp/exg_analysis)')
    p.add_argument('--majority', action='store_true', default=True,
                   help='Use majority vote for agreement (default: True)')
    p.add_argument('--unanimous', dest='majority', action='store_false',
                   help='Require unanimity instead of majority vote')
    p.add_argument('--delta', action='store_true',
                   help='Also analyse ExG-delta (FP ExG − control ExG) if available')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    n_labellers = len(args.files)
    names = (args.names if args.names and len(args.names) == n_labellers
             else [f'Labeller{i+1}' for i in range(n_labellers)])

    print(f'Loading {n_labellers} labeller file(s) ...')
    all_records: List[List[Dict]] = []
    for path, name in zip(args.files, names):
        recs = load_file(path)
        print(f'  {name}: {len(recs)} instances from {path}')
        all_records.append(recs)

    merged = merge_labellers(all_records, names)
    print(f'\nTotal unique instances (matched): {len(merged)}')

    seen_by_multiple = sum(1 for inst in merged.values() if len(inst['votes']) > 1)
    print(f'  Seen by 2+ labellers: {seen_by_multiple}')
    if seen_by_multiple < 5:
        print('  WARNING: very few overlapping instances. '
              'Instances from all labellers will be pooled (each labeller '
              'is treated as an independent vote).')

    plants, nonplants, disputed = classify_instances(
        merged, majority=args.majority)

    plant_exg    = np.array([i['exg'] for i in plants    if i['exg'] is not None])
    nonplant_exg = np.array([i['exg'] for i in nonplants if i['exg'] is not None])
    disputed_exg = np.array([i['exg'] for i in disputed  if i['exg'] is not None])

    if len(plant_exg) == 0 or len(nonplant_exg) == 0:
        print('\nERROR: not enough agreed instances in one class to compute a threshold.')
        print('  Agreed plant:     ', len(plant_exg))
        print('  Agreed not-plant: ', len(nonplant_exg))
        return

    thresholds, accuracies, best_thr, best_acc = threshold_sweep(
        plant_exg, nonplant_exg)
    midpoint_thr = (plant_exg.mean() + nonplant_exg.mean()) / 2.0

    # Per-labeller breakdown for plot markers
    labeller_data: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for labeller_recs, name in zip(all_records, names):
        lp = np.array([r['exg'] for r in labeller_recs
                       if r['exg'] is not None and r['is_plant']])
        ln = np.array([r['exg'] for r in labeller_recs
                       if r['exg'] is not None and not r['is_plant']])
        labeller_data.append((name, lp, ln))

    print_stats(plants, nonplants, disputed, labeller_data,
                best_thr, best_acc, midpoint_thr)

    fig_path = osp.join(args.output, 'exg_threshold_analysis.png')
    plot_analysis(plant_exg, nonplant_exg, disputed_exg,
                  labeller_data, thresholds, accuracies,
                  best_thr, best_acc, midpoint_thr, fig_path)

    # Optionally analyse ExG-delta (FP ExG − control background ExG)
    if args.delta:
        delta_plant    = np.array([i['exg_delta'] for i in plants
                                   if i.get('exg_delta') is not None])
        delta_nonplant = np.array([i['exg_delta'] for i in nonplants
                                   if i.get('exg_delta') is not None])
        if len(delta_plant) > 1 and len(delta_nonplant) > 1:
            dt, da, db_thr, db_acc = threshold_sweep(delta_plant, delta_nonplant)
            mid_d = (delta_plant.mean() + delta_nonplant.mean()) / 2.0
            delta_labeller = []
            for labeller_recs, name in zip(all_records, names):
                lp = np.array([r['exg_delta'] for r in labeller_recs
                               if r.get('exg_delta') is not None and r['is_plant']])
                ln = np.array([r['exg_delta'] for r in labeller_recs
                               if r.get('exg_delta') is not None and not r['is_plant']])
                delta_labeller.append((name, lp, ln))
            delta_fig = osp.join(args.output, 'exg_delta_threshold_analysis.png')
            print(f'\n--- ExG Delta (FP − background control) ---')
            print(f'  Delta plant    mean={delta_plant.mean():+.2f}  std={delta_plant.std():.2f}')
            print(f'  Delta nonplant mean={delta_nonplant.mean():+.2f}  std={delta_nonplant.std():.2f}')
            print(f'  Midpoint thr  = {mid_d:+.2f}')
            print(f'  Optimal thr   = {db_thr:+.2f}  ({db_acc*100:.1f}% acc)')
            plot_analysis(delta_plant, delta_nonplant,
                          np.array([i['exg_delta'] for i in disputed
                                    if i.get('exg_delta') is not None]),
                          delta_labeller, dt, da, db_thr, db_acc, mid_d,
                          delta_fig)

    # Save merged results to CSV
    merged_csv = osp.join(args.output, 'merged_labels.csv')
    os.makedirs(args.output, exist_ok=True)
    with open(merged_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = (['img_name', 'date_prefix', 'area', 'exg', 'ctrl_exg',
                       'exg_delta', 'agreement']
                      + [f'vote_{n}' for n in names])
        w = __import__('csv').DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        key_to_class = {}
        for inst in plants:
            key_to_class[id(inst)] = 'plant'
        for inst in nonplants:
            key_to_class[id(inst)] = 'not-plant'
        for inst in disputed:
            key_to_class[id(inst)] = 'disputed'
        all_insts = plants + nonplants + disputed
        for inst in all_insts:
            row = {
                'img_name':   inst['img_name'],
                'date_prefix': inst['date_prefix'],
                'area':       inst['area'],
                'exg':        inst['exg'],
                'ctrl_exg':   inst.get('ctrl_exg'),
                'exg_delta':  inst.get('exg_delta'),
                'agreement':  key_to_class[id(inst)],
            }
            for name in names:
                row[f'vote_{name}'] = inst['votes'].get(name, '')
            w.writerow(row)
    print(f'\nMerged labels saved: {merged_csv}')


if __name__ == '__main__':
    main()
