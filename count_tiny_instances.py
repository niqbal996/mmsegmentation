"""
Count tiny instances (< area_threshold px) per class, matching the behavior
of InstanceDetectionMetric: GT instances are counted regardless of whether
they touch the image border (border filtering only applies to predicted FPs).

Semantic class mapping (PhenoBenchReduceClasses):
  3 -> 1 (partial crop -> crop)
  4 -> 2 (partial weed -> weed)

Supported datasets:
  phenobench  : uint16 PNG instance maps, uint8 PNG semantics
  synthetic   : int64 NPZ instance maps, uint8 PNG semantics
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from multiprocessing import Pool
import os
from tqdm import tqdm

SEM_REMAP = {3: 1, 4: 2}
VALID_CLASSES = {1, 2}
CLASS_NAMES = {1: 'Crop', 2: 'Weed'}


def remap_sem(arr):
    out = arr.copy()
    for src, dst in SEM_REMAP.items():
        out[arr == src] = dst
    return out


def is_border_touching(mask):
    return mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()


# ── Phenobench ────────────────────────────────────────────────────────────────

def _process_pheno(args):
    inst_path, sem_path, area_thr, border_filter = args
    img_inst = np.array(Image.open(inst_path))
    img_sem = remap_sem(np.array(Image.open(sem_path)))

    result = {1: [0, 0], 2: [0, 0]}  # [total, tiny]
    for inst_id in np.unique(img_inst):
        if inst_id == 0:
            continue
        mask = img_inst == inst_id
        area = int(mask.sum())
        cls = int(np.bincount(img_sem[mask].astype(int)).argmax())
        if cls not in VALID_CLASSES:
            continue
        result[cls][0] += 1
        if area < area_thr:
            if not border_filter or not is_border_touching(mask):
                result[cls][1] += 1
    return result


def analyze_phenobench(data_root, splits, area_thr, border_filter, workers):
    data_root = Path(data_root)
    totals = {1: [0, 0], 2: [0, 0]}

    for split in splits:
        inst_dir = data_root / split / 'plant_instances'
        sem_dir = data_root / split / 'semantics'
        files = sorted(inst_dir.glob('*.png'))
        args = [(str(f), str(sem_dir / f.name), area_thr, border_filter)
                for f in files]

        print(f"\nPhenobench [{split}] — {len(files)} images")
        split_stats = {1: [0, 0], 2: [0, 0]}
        with Pool(workers) as pool:
            for res in tqdm(
                    pool.imap_unordered(_process_pheno, args, chunksize=20),
                    total=len(args), unit='img'):
                for cls in VALID_CLASSES:
                    split_stats[cls][0] += res[cls][0]
                    split_stats[cls][1] += res[cls][1]

        _print_split(split_stats, area_thr, border_filter)
        for cls in VALID_CLASSES:
            totals[cls][0] += split_stats[cls][0]
            totals[cls][1] += split_stats[cls][1]

    return totals


# ── SugarBeet Synthetic ───────────────────────────────────────────────────────

def _process_syn(args):
    sem_path, inst_dir, area_thr, border_filter = args
    stem = Path(sem_path).stem
    inst_path = Path(inst_dir) / f"{stem}.npz"
    if not inst_path.exists():
        return {1: [0, 0], 2: [0, 0]}

    img_inst = np.load(inst_path)['array']
    img_sem = remap_sem(np.array(Image.open(sem_path)))

    result = {1: [0, 0], 2: [0, 0]}
    for inst_id in np.unique(img_inst):
        mask = img_inst == inst_id
        area = int(mask.sum())
        cls = int(np.bincount(img_sem[mask].astype(int)).argmax())
        if cls not in VALID_CLASSES:
            continue
        result[cls][0] += 1
        if area < area_thr:
            if not border_filter or not is_border_touching(mask):
                result[cls][1] += 1
    return result


def analyze_synthetic(data_root, splits, area_thr, border_filter, workers):
    data_root = Path(data_root)
    inst_dir = data_root / 'main_camera_annotations' / 'instance_segmentation'
    totals = {1: [0, 0], 2: [0, 0]}

    for split in splits:
        sem_dir = data_root / 'main_camera_annotations' / 'semantics' / split
        files = sorted(sem_dir.glob('*.png'))
        args = [(str(f), str(inst_dir), area_thr, border_filter)
                for f in files]

        print(f"\nSugarBeet Synthetic [{split}] — {len(files)} images")
        split_stats = {1: [0, 0], 2: [0, 0]}
        with Pool(workers) as pool:
            for res in tqdm(
                    pool.imap_unordered(_process_syn, args, chunksize=20),
                    total=len(args), unit='img'):
                for cls in VALID_CLASSES:
                    split_stats[cls][0] += res[cls][0]
                    split_stats[cls][1] += res[cls][1]

        _print_split(split_stats, area_thr, border_filter)
        for cls in VALID_CLASSES:
            totals[cls][0] += split_stats[cls][0]
            totals[cls][1] += split_stats[cls][1]

    return totals


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_split(stats, area_thr, border_filter):
    border_note = ', non-border' if border_filter else ''
    for cls in (1, 2):
        t, tiny = stats[cls]
        pct = 100 * tiny / max(1, t)
        print(f"  {CLASS_NAMES[cls]:4s} — total: {t:7d}  "
              f"tiny(<{area_thr}px{border_note}): {tiny:6d} ({pct:.1f}%)")


def _print_totals(label, totals, area_thr, border_filter):
    border_note = ', non-border' if border_filter else ''
    print(f"\n{'─'*60}")
    print(f"  {label} TOTALS (all requested splits)")
    print(f"{'─'*60}")
    _print_split(totals, area_thr, border_filter)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Count instances per class with tiny-instance breakdown.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', choices=['phenobench', 'synthetic', 'both'],
                        default='both')
    parser.add_argument('--pheno-root', default='/mnt/e/datasets/phenobench')
    parser.add_argument('--syn-root', default='/mnt/e/datasets/sugarbeet_syn_v6')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help='Which splits to process')
    parser.add_argument('--area-thr', type=int, default=100,
                        help='Tiny instance pixel area threshold (exclusive)')
    parser.add_argument('--border-filter', action='store_true',
                        help='Exclude instances touching the image border from '
                             'tiny count (NOT done by InstanceDetectionMetric)')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    if args.dataset in ('phenobench', 'both'):
        totals = analyze_phenobench(
            args.pheno_root, args.splits, args.area_thr,
            args.border_filter, args.workers)
        _print_totals('PHENOBENCH', totals, args.area_thr, args.border_filter)

    if args.dataset in ('synthetic', 'both'):
        totals = analyze_synthetic(
            args.syn_root, args.splits, args.area_thr,
            args.border_filter, args.workers)
        _print_totals('SYNTHETIC', totals, args.area_thr, args.border_filter)


if __name__ == '__main__':
    main()
