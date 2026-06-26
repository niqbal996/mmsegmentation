#!/usr/bin/env python3
"""Generate a multi-run averaged LaTeX paper table from metrics_sweep_summary JSON files.

Loads 1-N JSON files, averages experiments that appear in >= --min-runs files,
and emits a complete LaTeX table to stdout (or --out file).

Usage:
    python generate_paper_table.py run1.json run2.json run3.json
    python generate_paper_table.py run*.json --min-runs 2 --out table.tex
    python generate_paper_table.py run1.json run2.json --n-images 772
"""

import argparse
import json
import re
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple


# ── 9 paper columns: (metric_key, higher_is_better) ──────────────────────────
COLS: List[Tuple[str, bool]] = [
    ('pixel_IoU_crop',              True),
    ('weed_iou',                    True),
    ('weed_precision',              True),
    ('weed_pixel_prec_le100',       True),
    ('obj_crop_le100_iog_recall',   True),
    ('obj_weed_le100_iog_recall',   True),
    ('_fp_img_crop',                False),
    ('_fp_img_soil_bare',           False),
    ('_fp_img_soil_veg',            True),   # higher = more detections on probable unlabelled veg
]
COL_KEYS = [k for k, _ in COLS]

# ── Model families (defines table section order) ──────────────────────────────
FAMILIES: List[Tuple[str, str]] = [
    ('deeplabv3plus', r'\textit{DeepLabV3+}'),
    ('segformer',     r'\textit{SegFormer}'),
    ('mask2former',   r'\textit{Mask2Former}'),
]

# ── Row order: (row_key, latex_label, group_id)
# group_id boundary → [2pt] spacer added after the last row of a group
ROWS: List[Tuple[str, str, int]] = [
    ('real_1pct',   r'Real 1\%   \;$(\mathcal{D}_{real}^{1\%})$',                              1),
    ('real_5pct',   r'Real 5\%   \;$(\mathcal{D}_{real}^{5\%})$',                              1),
    ('real_10pct',  r'Real 10\%  \;$(\mathcal{D}_{real}^{10\%})$',                             1),
    ('real_100',    r'Real 100\% \;$(\mathcal{D}_{real}^{100\%})$',                            1),
    ('mix_1pct',    r'Mix 1\%    \;$(\mathcal{D}_{syn} \cup \mathcal{D}_{real}^{1\%})$',       2),
    ('mix_5pct',    r'Mix 5\%    \;$(\mathcal{D}_{syn} \cup \mathcal{D}_{real}^{5\%})$',       2),
    ('mix_10pct',   r'Mix 10\%   \;$(\mathcal{D}_{syn} \cup \mathcal{D}_{real}^{10\%})$',      2),
    ('synth',       r'Synth (0\% real) \;$(\mathcal{D}_{syn})$',                               3),
]


# ── Classification ────────────────────────────────────────────────────────────

def _classify(name: str) -> Optional[Tuple[str, str]]:
    """Map experiment name → (family_key, row_key), or None to skip."""
    nl = name.lower()

    # Find dataset section — check longer markers first to avoid substring match
    dataset = model_part = variant = None
    for marker in (
        '_sugarbeetsynthetic2026_2phenobench',
        '_sugarbeetsynthetic2026',
        '_phenobench',
    ):
        idx = nl.find(marker)
        if idx < 0:
            continue
        model_part = nl[:idx]
        variant    = nl[idx + len(marker):].lstrip('_')
        dataset    = marker.lstrip('_')
        break

    if dataset is None:
        return None

    # Model family
    if model_part.startswith('deeplabv3plus'):
        family = 'deeplabv3plus'
    elif model_part.startswith('segformer'):
        family = 'segformer'
    elif model_part.startswith('mask2former') and 'swin' in model_part:
        family = 'mask2former'
    else:
        return None  # skip mask2former_r50, fcn, etc.

    # Row label
    m = re.search(r'real(\d+)pct', variant)
    pct = int(m.group(1)) if m else None

    if dataset == 'phenobench':
        row = {1: 'real_1pct', 5: 'real_5pct', 10: 'real_10pct'}.get(pct, 'real_100')
    elif dataset == 'sugarbeetsynthetic2026_2phenobench':
        row = {1: 'mix_1pct', 5: 'mix_5pct', 10: 'mix_10pct'}.get(pct, 'synth')
    else:
        # sugarbeetsynthetic2026 (standalone) — trained and validated on its own
        # synthetic test set, not on phenobench.  Skip entirely.
        return None

    return family, row


# ── Metric extraction ─────────────────────────────────────────────────────────

_OPTIONAL_METRICS = {'weed_pixel_prec_le100'}  # added later; absent in pre-release JSONs


def _extract(raw: Dict[str, Any],
             n_fallback: Optional[int]) -> Optional[Dict[str, Optional[float]]]:
    """Extract the 9 paper metric values from a raw metrics dict.

    Returns None only when the experiment is entirely unusable (missing FP keys
    or no n_images).  Optional metrics (weed_pixel_prec_le100) may be None when
    absent from older JSONs — the accumulator will skip those None values.
    """
    n_raw = raw.get('n_images')
    if n_raw is not None:
        n = max(1, int(n_raw))
    elif n_fallback is not None:
        n = max(1, n_fallback)
    else:
        return None  # cannot compute FP/img rates

    def g(k: str) -> Optional[float]:
        v = raw.get(k)
        return float(v) if v is not None else None

    fp_soil = g('obj_weed_le100_fp_on_background_soil')
    fp_veg  = g('obj_weed_le100_fp_probably_unlabelled')
    fp_crop = g('obj_weed_le100_fp_on_crop')
    if fp_soil is None or fp_veg is None or fp_crop is None:
        return None

    out: Dict[str, Optional[float]] = {
        'pixel_IoU_crop':            g('pixel_IoU_crop'),
        'weed_iou':                  g('weed_iou'),
        'weed_precision':            g('weed_precision'),
        'weed_pixel_prec_le100':     g('weed_pixel_prec_le100'),   # may be None
        'obj_crop_le100_iog_recall': g('obj_crop_le100_iog_recall'),
        'obj_weed_le100_iog_recall': g('obj_weed_le100_iog_recall'),
        '_fp_img_crop':              fp_crop / n,
        '_fp_img_soil_bare':         (fp_soil - fp_veg) / n,
        '_fp_img_soil_veg':          fp_veg / n,
    }
    if any(v is None for k, v in out.items() if k not in _OPTIONAL_METRICS):
        return None
    return out


# ── Loading ───────────────────────────────────────────────────────────────────

def load_run(path: str, n_fallback: Optional[int]) -> Dict[str, Dict[str, Optional[float]]]:
    """Load {experiment_name: paper_metrics} from one summary JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    exps = payload.get('experiments', {})
    items = list(exps.values()) if isinstance(exps, dict) else list(exps)

    run: Dict[str, Dict[str, Optional[float]]] = {}
    skipped = 0
    for item in items:
        if not isinstance(item, dict) or item.get('status') != 'success':
            continue
        name = item.get('experiment_name', '')
        raw  = item.get('metrics', {})
        if not isinstance(raw, dict):
            continue
        m = _extract(raw, n_fallback)
        if m is not None:
            run[name] = m
        else:
            skipped += 1

    if skipped:
        print(f'  [note] {skipped} experiments skipped (missing n_images or FP keys).',
              file=sys.stderr)
    return run


# ── Averaging ─────────────────────────────────────────────────────────────────

def _exp_priority(name: str) -> int:
    """Return selection priority for duplicate-cell resolution (higher wins).

    Priority is model-family dependent:
    - DeepLabV3+ / Mask2Former: prefer ohem_loss (more stable training).
    - SegFormer: prefer non-ohem_loss (OHEM is noisy for this architecture).
    """
    nl = name.lower()
    has_ohem = 'ohem_loss' in nl
    if nl.startswith('segformer'):
        return 1 if has_ohem else 2   # vanilla/baseline wins for SegFormer
    return 2 if has_ohem else 1       # ohem_loss wins for everything else


def average_runs(
    runs: List[Dict[str, Dict[str, float]]],
    min_runs: int,
) -> Dict[Tuple[str, str], Tuple[Dict[str, Tuple[float, float]], int]]:
    """
    Returns {(family_key, row_key): ({col_key: (mean, std)}, n_contributing_runs)}.
    Cells present in fewer than min_runs are excluded entirely.
    Cells present in ≥ min_runs but < total runs are included with their actual count,
    so callers can mark them as partial.

    When multiple experiments in the same run map to the same cell, the one with
    the highest _exp_priority() wins (ohem_loss > everything else).
    """
    # acc: cell → {col: [one value per run]}
    acc: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    # Track which experiment name was chosen for each (run_idx, cell)
    chosen: Dict[Tuple[int, Tuple[str, str]], Tuple[str, int]] = {}  # → (name, priority)

    for run_idx, run in enumerate(runs):
        for name, metrics in run.items():
            cls = _classify(name)
            if cls is None:
                continue

            priority = _exp_priority(name)
            run_cell = (run_idx, cls)

            if run_cell in chosen:
                prev_name, prev_priority = chosen[run_cell]
                if priority <= prev_priority:
                    print(f'  [skip] {cls} — keeping "{prev_name}" over "{name}" '
                          f'(priority {prev_priority} ≥ {priority})', file=sys.stderr)
                    continue
                # New experiment has higher priority — replace the previous contribution
                print(f'  [replace] {cls} — "{name}" (priority {priority}) '
                      f'replaces "{prev_name}" (priority {prev_priority})', file=sys.stderr)
                # Remove the last appended value for each column (from prev_name)
                for k in COL_KEYS:
                    if cls in acc and acc[cls][k]:
                        acc[cls][k].pop()

            chosen[run_cell] = (name, priority)
            if cls not in acc:
                acc[cls] = {k: [] for k in COL_KEYS}
            for k in COL_KEYS:
                v = metrics.get(k)
                if v is not None:
                    acc[cls][k].append(v)

    result: Dict[Tuple[str, str], Tuple[Dict[str, Tuple[float, float]], int]] = {}
    for cell, col_lists in acc.items():
        n = len(col_lists.get('weed_iou', []))
        if n < min_runs:
            continue
        cell_metrics: Dict[str, Tuple[float, float]] = {}
        for k, vals in col_lists.items():
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cell_metrics[k] = (mean, std)
        result[cell] = (cell_metrics, n)

    return result


# ── LaTeX formatting ──────────────────────────────────────────────────────────

def _cell(mean: float, std: float, level: int = 0) -> str:
    """Render a table cell.

    level 0 = plain
    level 1 = per-family best  → bold + underline
    level 2 = global best      → bold + underline + green (requires xcolor)
    """
    m_str = f'{mean:.2f}'
    if std > 0:
        math = f'${m_str}_{{\\pm{std:.2f}}}$'
        if level == 2:
            return f'\\textcolor{{ForestGreen}}{{\\underline{{\\boldmath${m_str}_{{\\pm{std:.2f}}}$}}}}'
        if level == 1:
            return f'\\underline{{\\boldmath${m_str}_{{\\pm{std:.2f}}}$}}'
        return math
    if level == 2:
        return f'\\textcolor{{ForestGreen}}{{\\underline{{\\textbf{{{m_str}}}}}}}'
    if level == 1:
        return f'\\underline{{\\textbf{{{m_str}}}}}'
    return m_str


def _best_vals(
    cells: Dict[Tuple[str, str], Tuple[Dict[str, Tuple[float, float]], int]],
    n_runs: int,
) -> Dict[str, float]:
    """Return {col_key: best_mean} preferring full-coverage cells."""
    result: Dict[str, float] = {}
    for k, higher in COLS:
        vals = [m[k][0] for m, n in cells.values() if k in m and n == n_runs]
        if not vals:
            vals = [m[k][0] for m, _ in cells.values() if k in m]
        if vals:
            result[k] = max(vals) if higher else min(vals)
    return result


def build_latex(
    averaged: Dict[Tuple[str, str], Tuple[Dict[str, Tuple[float, float]], int]],
    n_runs: int,
    label: str = 'tab:main',
) -> str:
    # Check whether any cell has partial coverage (fewer runs than total)
    has_partial = any(n < n_runs for _, n in averaged.values())

    # Global best across all families
    global_best = _best_vals(averaged, n_runs)

    # Per-family best
    family_best: Dict[str, Dict[str, float]] = {}
    for fam_key, _ in FAMILIES:
        fam_cells = {cell: v for cell, v in averaged.items() if cell[0] == fam_key}
        family_best[fam_key] = _best_vals(fam_cells, n_runs)

    def highlight_level(k: str, mean: float, fam_key: str) -> int:
        if k in global_best and abs(mean - global_best[k]) < 5e-3:
            return 2
        if k in family_best.get(fam_key, {}) and abs(mean - family_best[fam_key][k]) < 5e-3:
            return 1
        return 0

    run_note = (f'{n_runs}-run mean$_{{\\pm\\text{{std}}}}$'
                if n_runs > 1 else 'single run')
    partial_note = (r'~\textsuperscript{\dag}~averaged over fewer than '
                    + str(n_runs) + r' runs.')

    L: List[str] = []
    a = L.append

    a(r'\begin{table*}[t]')
    a(r'\centering')
    caption = (
        r'\caption{Evaluation on the PhenoBench \textbf{validation} set '
        r'($N{=}772$; official test set is server-side and label-hidden). '
        r'Values are ' + run_note + r'. '
        r'\textbf{Recall} is instance-level IoG\,$\geq\!0.05$ over tiny GT instances '
        r'($\leq\!100$\,px). \textbf{Precision} is pixel-area weed precision, '
        r'reported globally and over tiny predicted islands ($\leq\!100$\,px). '
        r'\textbf{FP/image} counts tiny weed predictions ($\leq\!100$\,px) per image, '
        r'split by landing surface: \emph{Crop}\,$\downarrow$ (safety-critical), '
        r'\emph{bare} soil\,$\downarrow$ (genuine false fire), and \emph{veg.}\ soil\,$\uparrow$ '
        r'(probable unlabelled plant, high ExG; higher means more sensitivity to real vegetation). '
        r'$\uparrow$/$\downarrow$: higher/lower is better. '
        r'\underline{\textbf{Bold+underline}}\,=\,best per model family; '
        r'\textcolor{ForestGreen}{\underline{\textbf{green}}}\,=\,global best across all models'
        + (r' (full-coverage rows only).' + partial_note if has_partial else r'.')
        + r'}'
    )
    a(caption)
    a(f'\\label{{{label}}}')
    a(r'\setlength{\tabcolsep}{4pt}')
    a(r'\resizebox{\textwidth}{!}{%')
    a(r'\begin{tabular}{@{}l cc cc cc ccc@{}}')
    a(r'\toprule')
    a(r' & \multicolumn{2}{c}{\textbf{Semantic IoU (\%)}\,$\uparrow$}'
      r' & \multicolumn{2}{c}{\textbf{Weed Pixel Prec.\ (\%)}\,$\uparrow$}'
      r' & \multicolumn{2}{c}{\textbf{Tiny Recall (IoG, \%)}\,$\uparrow$}'
      r' & \multicolumn{3}{c}{\textbf{Tiny-Weed FP\,/\,image}} \\')
    a(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-10}')
    a(r'\textbf{Train Data}'
      r' & \textbf{Crop} & \textbf{Weed}'
      r' & \textbf{All} & \textbf{$\leq$100px}'
      r' & \textbf{Crop} & \textbf{Weed}'
      r' & \textbf{Crop}\,$\downarrow$ & \textbf{Soil$_{\text{bare}}$}\,$\downarrow$'
      r' & \textbf{Soil$_{\text{veg}}$}\,$\uparrow$ \\')

    for fam_key, fam_display in FAMILIES:
        a(r'\midrule')
        a(f'\\multicolumn{{10}}{{@{{}}l}}{{\\textbf{{{fam_display}}}}} \\\\')
        a(r'\midrule')

        present = [(rk, rl, grp)
                   for rk, rl, grp in ROWS
                   if (fam_key, rk) in averaged]

        for i, (row_key, row_label, grp) in enumerate(present):
            cell_metrics, n_cell = averaged[(fam_key, row_key)]

            # Mark partial rows with a dagger superscript
            if n_cell < n_runs:
                label_tex = row_label + r'\textsuperscript{\dag}'
            else:
                label_tex = row_label

            cells = [label_tex]
            for k, _ in COLS:
                if k not in cell_metrics:
                    cells.append('--')
                else:
                    mean, std = cell_metrics[k]
                    cells.append(_cell(mean, std, level=highlight_level(k, mean, fam_key)))

            spacer = ''
            if i + 1 < len(present) and present[i + 1][2] != grp:
                spacer = '[2pt]'

            a(' & '.join(cells) + f' \\\\{spacer}')

    a(r'\bottomrule')
    a(r'\end{tabular}%')
    a(r'}')
    a(r'\end{table*}')

    return '\n'.join(L)


# ── ASCII preview ─────────────────────────────────────────────────────────────

def print_ascii_preview(
    averaged: Dict[Tuple[str, str], Tuple[Dict[str, Tuple[float, float]], int]],
    n_total: int = 0,
) -> None:
    col_headers = [
        'Experiment', 'IoU Crop', 'IoU Weed', 'Prec All', 'Prec<=100',
        'Rec Crop', 'Rec Weed', 'FP Crop', 'FP Bare', 'FP Veg',
    ]
    rows: List[List[str]] = []
    for fam_key, _ in FAMILIES:
        for row_key, row_label, _ in ROWS:
            cell = (fam_key, row_key)
            if cell not in averaged:
                continue
            cell_metrics, n_cell = averaged[cell]
            suffix = f' ({n_cell}/{n_total})' if (n_total > 1 and n_cell < n_total) else ''
            exp_label = f'{fam_key[:3].upper()} {row_label}{suffix}'
            vals = [exp_label]
            for k, _ in COLS:
                if k not in cell_metrics:
                    vals.append('-')
                else:
                    mean, std = cell_metrics[k]
                    if std > 0:
                        vals.append(f'{mean:.2f}±{std:.2f}')
                    else:
                        vals.append(f'{mean:.2f}')
            rows.append(vals)

    if not rows:
        print('No data to display.', file=sys.stderr)
        return

    widths = [max(len(col_headers[i]), *(len(r[i]) for r in rows))
              for i in range(len(col_headers))]
    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    hdr = '| ' + ' | '.join(h.ljust(widths[i]) for i, h in enumerate(col_headers)) + ' |'

    print('\nASCII Preview')
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print('| ' + ' | '.join(v.ljust(widths[i]) for i, v in enumerate(row)) + ' |')
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Generate a LaTeX paper table averaged across multiple evaluation runs.')
    p.add_argument(
        'json_files', nargs='+',
        help='Path(s) to metrics_sweep_summary.json (one per run).')
    p.add_argument(
        '--min-runs', type=int, default=1,
        help=(
            'Minimum runs a cell must appear in to be included at all (default: 1). '
            'Cells with fewer than the total number of input files are shown with a '
            'dagger (†) marker. Use --min-runs N to exclude partial cells entirely.'))
    p.add_argument(
        '--n-images', type=int, default=None,
        help='Fallback n_images for JSONs that predate the n_images metric (e.g. 772).')
    p.add_argument(
        '--out', default=None,
        help='Write LaTeX to this file (default: stdout).')
    p.add_argument(
        '--label', default='tab:main',
        help='LaTeX \\label value (default: tab:main).')
    p.add_argument(
        '--preview', action='store_true',
        help='Also print an ASCII preview table to stderr.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_total = len(args.json_files)
    min_runs = args.min_runs

    runs: List[Dict[str, Dict[str, float]]] = []
    for path in args.json_files:
        print(f'Loading {path} ...', file=sys.stderr)
        run = load_run(path, args.n_images)
        runs.append(run)
        print(f'  {len(run)} classifiable experiments loaded.', file=sys.stderr)

    averaged = average_runs(runs, min_runs)
    n_full = sum(1 for _, n in averaged.values() if n == n_total)
    n_partial = len(averaged) - n_full
    print(
        f'\nCells: {n_full} complete ({n_total}/{n_total} runs)'
        + (f', {n_partial} partial (†)' if n_partial else '')
        + f', {len(set(c for r in runs for c in [_classify(nm) for nm in r] if c)) - len(averaged)} excluded (< min_runs={min_runs}).',
        file=sys.stderr)

    if args.preview:
        print_ascii_preview(averaged, n_total=n_total)

    tex = build_latex(averaged, n_runs=n_total, label=args.label)

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(tex + '\n')
        print(f'\nLaTeX written to: {args.out}', file=sys.stderr)
    else:
        print(tex)


if __name__ == '__main__':
    main()
