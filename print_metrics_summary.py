#!/usr/bin/env python3
"""Pretty-print a metrics sweep summary without rerunning evaluation."""

import argparse
import json
import os.path as osp
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_SUMMARY = 'metrics_sweep_summary.json'
PREFERRED_METRICS = ['mIoU', 'mAcc', 'aAcc', 'IoU_weed', 'Acc_weed', 'time']


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


def model_dataset_sort_key(item: Dict[str, Any]) -> tuple:
    name = experiment_name(item)
    dataset_order = {
        'phenobench': 0,
        'sugarbeetsynthetic2026': 1,
        'sugarbeetsynthetic2026_2phenobench': 2,
    }
    variant_order = {
        'baseline': 0,
        'vanilla': 0,
        '': 0,
        'ohem_loss': 1,
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
            variant_order.get(variant, 10),
            variant.lower(),
            name.lower(),
        )

    return (name.lower(), len(dataset_order), 10, '', name.lower())


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


if __name__ == '__main__':
    main()
