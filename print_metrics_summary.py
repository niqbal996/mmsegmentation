#!/usr/bin/env python3
"""Pretty-print a metrics sweep summary without rerunning evaluation."""

import argparse
import json
import math
import os.path as osp
import re
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_SUMMARY = 'metrics_sweep_summary.json'
PREFERRED_METRICS = ['mIoU', 'mAcc', 'aAcc', 'IoU_weed', 'Acc_weed', 'time']

_PAPER_COLS = [
    'IoU Crop(%)', 'IoU Weed(%)', 'WeedPrec All(%)', 'WeedPrec<=100px(%)',
    'TinyRec Crop(%)', 'TinyRec Weed(%)', 'FP/img Crop', 'FP/img Soil_bare', 'FP/img Soil_veg',
]


def _paper_row(metrics: Dict[str, Any], n_images_fallback: Optional[int] = None) -> List[str]:
    stored = metrics.get('n_images')
    if stored is not None:
        n = max(1, int(stored))
    elif n_images_fallback is not None:
        n = max(1, n_images_fallback)
    else:
        n = None  # unknown — show raw counts with a '/' marker

    fp_soil = metrics.get('obj_weed_le100_fp_on_background_soil', float('nan'))
    fp_veg = metrics.get('obj_weed_le100_fp_probably_unlabelled', float('nan'))
    fp_bare = (fp_soil - fp_veg
               if not (math.isnan(float(fp_soil)) or math.isnan(float(fp_veg)))
               else float('nan'))
    fp_crop = metrics.get('obj_weed_le100_fp_on_crop', float('nan'))

    def _f(v: Any) -> str:
        if v is None:
            return '-'
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return '-'
        if math.isnan(fv):
            return '-'
        return str(round(fv, 2))

    def _fp(raw: Any) -> str:
        if n is None:
            # No n_images available: show raw count with a note
            v = _f(raw)
            return v + '(raw)' if v != '-' else '-'
        try:
            return _f(float(raw) / n)
        except (TypeError, ValueError):
            return '-'

    return [
        _f(metrics.get('pixel_IoU_crop')),
        _f(metrics.get('weed_iou')),
        _f(metrics.get('weed_precision')),
        _f(metrics.get('weed_pixel_prec_le100')),
        _f(metrics.get('obj_crop_le100_iog_recall')),
        _f(metrics.get('obj_weed_le100_iog_recall')),
        _fp(fp_crop),
        _fp(fp_bare),
        _fp(fp_veg),
    ]


def print_paper_table(results: Sequence[Dict[str, Any]],
                      n_images_fallback: Optional[int] = None) -> None:
    successful = [item for item in results if item.get('status') == 'success']
    if not successful:
        print('\nWACV Paper Table\nNo successful experiments found.')
        return

    columns = ['Experiment'] + _PAPER_COLS
    rows: List[Dict[str, str]] = []
    for item in successful:
        m = item.get('metrics', {})
        row: Dict[str, str] = {'Experiment': experiment_name(item)}
        for col, val in zip(_PAPER_COLS,
                            _paper_row(m if isinstance(m, dict) else {},
                                       n_images_fallback=n_images_fallback)):
            row[col] = val
        rows.append(row)

    widths = {col: max(len(col), *(len(r[col]) for r in rows)) for col in columns}
    sep = '+' + '+'.join('-' * (widths[c] + 2) for c in columns) + '+'
    hdr = '| ' + ' | '.join(c.ljust(widths[c]) for c in columns) + ' |'

    print('\nWACV Paper Table')
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print('| ' + ' | '.join(row[c].ljust(widths[c]) for c in columns) + ' |')
    print(sep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Print metrics from an existing metrics_sweep_summary.json')
    parser.add_argument(
        '--summary-output',
        '--summary',
        default=DEFAULT_SUMMARY,
        help='Path to the consolidated metrics summary JSON')
    parser.add_argument(
        '--columns',
        default=None,
        help=(
            'Comma-separated metric columns to print. By default, print every '
            'metric present in the summary, matching generate_metrics.py.'))
    parser.add_argument(
        '--sort-by',
        default=None,
        help=(
            'Metric key used to sort successful experiments before printing. '
            'Use "model" to group by model/backbone and dataset order.'))
    parser.add_argument(
        '--ascending',
        action='store_true',
        help='Sort ascending instead of descending when --sort-by is set')
    parser.add_argument(
        '--successful-only',
        action='store_true',
        help='Do not print the failed experiments section')
    parser.add_argument(
        '--paper-table',
        action='store_true',
        help='Print the WACV 9-column paper table instead of (or in addition to) the full table')
    parser.add_argument(
        '--n-images',
        type=int,
        default=None,
        help=(
            'Number of validation images used to normalise FP/img columns in '
            'the paper table. Only needed for JSON files generated before '
            'n_images was saved (i.e. older runs). New runs store it '
            'automatically. Example: --n-images 772'))
    return parser.parse_args()


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f'Expected a JSON object in {path}')
    return payload


def results_from_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    experiments = summary.get('experiments', {})

    if isinstance(experiments, dict):
        results = list(experiments.values())
    elif isinstance(experiments, list):
        results = experiments
    else:
        raise ValueError('Summary field "experiments" must be a dict or list')

    normalized: List[Dict[str, Any]] = []
    for item in results:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized


def ordered_metric_keys(results: Sequence[Dict[str, Any]]) -> List[str]:
    all_keys = set()

    for result in results:
        metrics = result.get('metrics', {})
        if isinstance(metrics, dict):
            all_keys.update(metrics.keys())

    output: List[str] = []
    for key in PREFERRED_METRICS:
        if key in all_keys:
            output.append(key)
            all_keys.remove(key)

    output.extend(sorted(all_keys))
    return output


def parse_columns(columns: Optional[str],
                  results: Sequence[Dict[str, Any]]) -> List[str]:
    if columns is None:
        return ordered_metric_keys(results)

    parsed = [item.strip() for item in columns.split(',') if item.strip()]
    if not parsed:
        raise ValueError('--columns was provided but no metric names were found')
    return parsed


def format_metric_value(value: Any) -> str:
    if value is None:
        return '-'
    if isinstance(value, Number) and not isinstance(value, bool):
        text = f'{float(value):.4f}'
        text = text.rstrip('0').rstrip('.')
        return text if text else '0'
    return str(value)


def metric_sort_value(item: Dict[str, Any], key: str) -> float:
    metrics = item.get('metrics', {})
    value = metrics.get(key) if isinstance(metrics, dict) else None
    if isinstance(value, Number) and not isinstance(value, bool):
        return float(value)
    return float('-inf')


def experiment_name(item: Dict[str, Any]) -> str:
    name = item.get('experiment_name')
    if isinstance(name, str) and name:
        return name
    return '?'


def _variant_pct(variant: str) -> float:
    """Return the data-percentage for a variant string.

    Variants with an explicit NNpct marker sort by that number (1, 5, 10, …).
    Baseline / ohem_loss / vanilla / empty variants are treated as 100 % and
    sort last.
    """
    m = re.search(r'(\d+)pct', variant)
    if m:
        return float(m.group(1))
    return 100.0


def model_dataset_sort_key(item: Dict[str, Any]) -> tuple:
    name = experiment_name(item)
    # Longer dataset names must be checked first to avoid '_phenobench'
    # matching inside '_sugarbeetsynthetic2026_2phenobench'.
    dataset_order = {
        'sugarbeetsynthetic2026_2phenobench': 2,
        'sugarbeetsynthetic2026': 1,
        'phenobench': 0,
    }

    for dataset, dataset_idx in dataset_order.items():
        marker = f'_{dataset}'
        if marker not in name:
            continue

        model_part, variant = name.split(marker, 1)
        variant = variant.lstrip('_')
        return (
            model_part.lower(),
            dataset_idx,
            _variant_pct(variant),
            variant.lower(),
            name.lower(),
        )

    return (name.lower(), len(dataset_order), 100.0, '', name.lower())


def sort_results(results: Sequence[Dict[str, Any]],
                 sort_by: Optional[str],
                 ascending: bool) -> List[Dict[str, Any]]:
    if sort_by is None:
        return list(results)

    if sort_by in {'model', 'model_name', 'model-name', 'experiment'}:
        return sorted(results, key=model_dataset_sort_key)

    return sorted(
        results,
        key=lambda item: metric_sort_value(item, sort_by),
        reverse=not ascending)


def print_table(results: Sequence[Dict[str, Any]],
                metric_keys: Sequence[str],
                successful_only: bool = False) -> None:
    successful = [item for item in results if item.get('status') == 'success']
    failed = [item for item in results if item.get('status') != 'success']

    if successful:
        columns = ['experiment'] + list(metric_keys)

        rows: List[Dict[str, str]] = []
        for item in successful:
            metrics = item.get('metrics', {})
            row: Dict[str, str] = {
                'experiment': experiment_name(item)
            }
            for key in metric_keys:
                value = metrics.get(key) if isinstance(metrics, dict) else None
                row[key] = format_metric_value(value)
            rows.append(row)

        widths = {
            col: max(len(col), *(len(row[col]) for row in rows))
            for col in columns
        }

        separator = '+' + '+'.join('-' * (widths[c] + 2) for c in columns) + '+'
        header = '| ' + ' | '.join(c.ljust(widths[c]) for c in columns) + ' |'

        print('\nEvaluation Metrics')
        print(separator)
        print(header)
        print(separator)
        for row in rows:
            print('| ' + ' | '.join(row[c].ljust(widths[c]) for c in columns) +
                  ' |')
        print(separator)
    else:
        print('\nEvaluation Metrics')
        print('No successful experiments with metrics were found.')

    if failed and not successful_only:
        print('\nFailed Experiments')
        for item in failed:
            print(f"- {item.get('experiment_name', '?')}: "
                  f"{item.get('error', 'Unknown error')}")


def main() -> None:
    args = parse_args()
    summary_path = osp.abspath(args.summary_output)
    summary = load_summary(summary_path)
    results = results_from_summary(summary)

    successful = [item for item in results if item.get('status') == 'success']
    metric_keys = parse_columns(args.columns, successful)

    results = sort_results(results, args.sort_by, args.ascending)

    print(f'Loaded metrics summary: {summary_path}')
    if 'generated_at' in summary:
        print(f"Generated at: {summary['generated_at']}")
    print(
        f"Experiments: {len(results)} total, "
        f"{sum(1 for item in results if item.get('status') == 'success')} "
        f"successful, "
        f"{sum(1 for item in results if item.get('status') != 'success')} "
        f"failed")

    print_table(results, metric_keys, successful_only=args.successful_only)
    if args.paper_table:
        print_paper_table(results, n_images_fallback=args.n_images)


if __name__ == '__main__':
    main()
