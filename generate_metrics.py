#!/usr/bin/env python3
"""Sweep scheduled MMSeg experiments and aggregate evaluation metrics.

This script parses schedule_trainings.sh, finds each experiment's best checkpoint,
runs test in MMSeg fashion, writes a numbered metrics json in a per-workdir
subfolder, and saves a consolidated summary json across all experiments.
"""

import argparse
import datetime as dt
import json
import logging
import os
import os.path as osp
import re
import subprocess
import time
from glob import glob
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

register_all_modules()

SCHEDULE_ENTRY_PATTERN = re.compile(
    r'^\s*"(?P<config>[^";]+);(?P<workdir>[^"]+)"\s*,?\s*$')
TIMESTAMP_DIR_PATTERN = re.compile(r'^\d{8}_\d{6}$')
METRICS_FILE_PATTERN = re.compile(r'^(?P<idx>\d+)_metrics\.json$')
ITER_PATTERN = re.compile(r'_iter_(\d+)\.pth$')
NUMBER_PATTERN = re.compile(r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate metrics for experiments from schedule_trainings.sh')
    parser.add_argument(
        '--schedule-file',
        default='schedule_trainings.sh',
        help='Path to schedule_trainings.sh')
    parser.add_argument(
        '--metrics-subdir',
        default='metrics_reports',
        help='Subdirectory created inside each work_dir for numbered metric jsons')
    parser.add_argument(
        '--summary-output',
        default='metrics_sweep_summary.json',
        help='Path to write consolidated json summary')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher, same semantics as tools/test.py')
    parser.add_argument(
        '--tta', action='store_true', help='Enable test-time augmentation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options in key=value format')
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help=('If set, predictions are saved for offline evaluation. '
              'Each experiment writes into <out>/<experiment_name>.'))
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Stop immediately when one experiment fails')
    parser.add_argument(
        '--local_rank', '--local-rank', type=int, default=0,
        help='Local rank for distributed launchers')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def normalize_path(path: str, base_dir: str) -> str:
    if osp.isabs(path):
        return osp.normpath(path)
    return osp.normpath(osp.join(base_dir, path))


def parse_schedule_file(schedule_file: str) -> List[Tuple[str, str]]:
    if not osp.isfile(schedule_file):
        raise FileNotFoundError(f'Schedule file not found: {schedule_file}')

    base_dir = osp.dirname(osp.abspath(schedule_file))
    entries: List[Tuple[str, str]] = []

    with open(schedule_file, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            match = SCHEDULE_ENTRY_PATTERN.match(line)
            if not match:
                continue

            config_raw = match.group('config').strip()
            workdir_raw = match.group('workdir').strip()
            config_path = normalize_path(config_raw, base_dir)
            work_dir = normalize_path(workdir_raw, base_dir)
            entries.append((config_path, work_dir))

    # New scheduler format builds entries dynamically from CONFIG_FILES.
    # If no legacy "config;work_dir" entries were found, ask the scheduler to
    # print resolved pairs and parse those.
    if entries:
        return entries

    try:
        proc = subprocess.run(
            ['bash', schedule_file, '--print-trainings'],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False)
    except OSError:
        return entries

    if proc.returncode != 0:
        return entries

    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('Printed '):
            continue
        if ';' not in line:
            continue

        config_raw, workdir_raw = line.split(';', 1)
        config_path = normalize_path(config_raw.strip(), base_dir)
        work_dir = normalize_path(workdir_raw.strip(), base_dir)
        entries.append((config_path, work_dir))

    return entries


def experiment_name_from_work_dir(work_dir: str) -> str:
    return osp.basename(osp.normpath(work_dir))


def parse_iter_from_ckpt(path: str) -> int:
    match = ITER_PATTERN.search(osp.basename(path))
    if not match:
        return -1
    return int(match.group(1))


def find_best_checkpoint(work_dir: str) -> Optional[str]:
    primary = glob(osp.join(work_dir, 'best_mIoU_iter_*.pth'))
    fallback = glob(osp.join(work_dir, 'best_*_iter_*.pth'))
    candidates = primary if primary else fallback

    if not candidates:
        return None

    candidates.sort(key=lambda p: (parse_iter_from_ckpt(p), p), reverse=True)
    return candidates[0]


def list_timestamp_dirs(work_dir: str) -> List[str]:
    if not osp.isdir(work_dir):
        return []

    dirs: List[str] = []
    for item in os.listdir(work_dir):
        path = osp.join(work_dir, item)
        if osp.isdir(path) and TIMESTAMP_DIR_PATTERN.match(item):
            dirs.append(path)

    dirs.sort()
    return dirs


def latest_timestamp_dir(work_dir: str) -> Optional[str]:
    dirs = list_timestamp_dirs(work_dir)
    if not dirs:
        return None
    return dirs[-1]


def trigger_tta(cfg: Config) -> None:
    cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    cfg.tta_model.module = cfg.model
    cfg.model = cfg.tta_model


def to_builtin_scalar(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, Number):
        return float(value)

    item = getattr(value, 'item', None)
    if callable(item):
        try:
            out = item()
            if isinstance(out, Number) and not isinstance(out, bool):
                return float(out)
        except Exception:
            return None

    return None


def normalize_metrics_dict(metrics: Any) -> Dict[str, float]:
    if not isinstance(metrics, dict):
        return {}

    normalized: Dict[str, float] = {}
    for key, value in metrics.items():
        scalar = to_builtin_scalar(value)
        if scalar is not None:
            normalized[str(key)] = scalar
    return normalized


def try_load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if not content:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    last_obj = None
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last_obj = json.loads(line)
        except json.JSONDecodeError:
            continue
    return last_obj


def select_best_metric_payload(payload: Any) -> Dict[str, float]:
    if isinstance(payload, dict):
        if 'metrics' in payload and isinstance(payload['metrics'], dict):
            candidate = normalize_metrics_dict(payload['metrics'])
            if candidate:
                return candidate

        direct = normalize_metrics_dict(payload)
        if direct:
            return direct

        best: Dict[str, float] = {}
        for value in payload.values():
            nested = select_best_metric_payload(value)
            if len(nested) > len(best):
                best = nested
        return best

    if isinstance(payload, list):
        best = {}
        for item in payload:
            nested = select_best_metric_payload(item)
            if len(nested) > len(best):
                best = nested
        return best

    return {}


def extract_metrics_from_latest_dir(test_dir: Optional[str]) -> Dict[str, float]:
    if test_dir is None or not osp.isdir(test_dir):
        return {}

    best_metrics: Dict[str, float] = {}
    json_files = sorted(glob(osp.join(test_dir, '**', '*.json'), recursive=True))

    for json_path in json_files:
        try:
            payload = try_load_json(json_path)
        except OSError:
            continue

        metrics = select_best_metric_payload(payload)
        if len(metrics) > len(best_metrics):
            best_metrics = metrics

    return best_metrics


def _to_float(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    if not NUMBER_PATTERN.match(text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def extract_classwise_metrics_from_log(test_dir: Optional[str]) -> Dict[str, float]:
    """Parse class-wise metrics table from mmseg test logs.

    Expected table section in logs:
      per class results:
      +------+-----+-----+
      | Class| IoU | Acc |
      | weed | 68.9| 89.3|
    """
    if test_dir is None or not osp.isdir(test_dir):
        return {}

    log_files = sorted(glob(osp.join(test_dir, '*.log')), key=osp.getmtime)
    if not log_files:
        return {}

    log_path = log_files[-1]
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except OSError:
        return {}

    last_marker = -1
    for idx, line in enumerate(lines):
        if 'per class results:' in line:
            last_marker = idx

    if last_marker < 0:
        return {}

    table_lines: List[str] = []
    for line in lines[last_marker + 1:]:
        if '+' in line or '|' in line:
            table_lines.append(line.strip())
            continue
        if table_lines:
            break

    rows: List[List[str]] = []
    for line in table_lines:
        if not line.startswith('|'):
            continue
        parts = [p.strip() for p in line.strip().strip('|').split('|')]
        if parts:
            rows.append(parts)

    if len(rows) < 2:
        return {}

    header = rows[0]
    try:
        class_idx = header.index('Class')
    except ValueError:
        return {}

    metric_cols: List[Tuple[int, str]] = []
    for i, col_name in enumerate(header):
        if i == class_idx:
            continue
        if not col_name:
            continue
        metric_cols.append((i, col_name))

    classwise: Dict[str, float] = {}
    for row in rows[1:]:
        if class_idx >= len(row):
            continue
        class_name = row[class_idx].replace(' ', '_')
        if not class_name:
            continue

        for col_idx, metric_name in metric_cols:
            if col_idx >= len(row):
                continue
            value = _to_float(row[col_idx])
            if value is None:
                continue
            classwise[f'{metric_name}_{class_name}'] = value

    return classwise


def next_metrics_index(metrics_dir: str) -> int:
    if not osp.isdir(metrics_dir):
        return 1

    max_idx = 0
    for filename in os.listdir(metrics_dir):
        match = METRICS_FILE_PATTERN.match(filename)
        if not match:
            continue
        max_idx = max(max_idx, int(match.group('idx')))

    return max_idx + 1


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(osp.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def ordered_metric_keys(results: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = ['mIoU', 'mAcc', 'aAcc', 'IoU_weed', 'Acc_weed', 'time']
    all_keys = set()

    for result in results:
        metrics = result.get('metrics', {})
        if isinstance(metrics, dict):
            all_keys.update(metrics.keys())

    output: List[str] = []
    for key in preferred:
        if key in all_keys:
            output.append(key)
            all_keys.remove(key)

    output.extend(sorted(all_keys))
    return output


def format_metric_value(value: Any) -> str:
    if value is None:
        return '-'
    if isinstance(value, Number) and not isinstance(value, bool):
        text = f'{float(value):.4f}'
        text = text.rstrip('0').rstrip('.')
        return text if text else '0'
    return str(value)


def print_table(results: Sequence[Dict[str, Any]]) -> None:
    successful = [item for item in results if item.get('status') == 'success']
    failed = [item for item in results if item.get('status') != 'success']

    if successful:
        metric_keys = ordered_metric_keys(successful)
        columns = ['experiment'] + metric_keys

        rows: List[Dict[str, str]] = []
        for item in successful:
            metrics = item.get('metrics', {})
            row: Dict[str, str] = {'experiment': item.get('experiment_name', '?')}
            for key in metric_keys:
                row[key] = format_metric_value(metrics.get(key))
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
            print('| ' + ' | '.join(row[c].ljust(widths[c]) for c in columns) + ' |')
        print(separator)

    if failed:
        print('\nFailed Experiments')
        for item in failed:
            print(f"- {item.get('experiment_name', '?')}: {item.get('error', 'Unknown error')}")


def run_single_experiment(config_path: str, work_dir: str,
                          args: argparse.Namespace) -> Dict[str, Any]:
    experiment_name = experiment_name_from_work_dir(work_dir)
    result: Dict[str, Any] = {
        'experiment_name': experiment_name,
        'config': config_path,
        'work_dir': work_dir,
        'status': 'failed',
    }

    if not osp.isfile(config_path):
        result['error'] = f'Config file not found: {config_path}'
        return result

    best_ckpt = find_best_checkpoint(work_dir)
    if best_ckpt is None:
        result['error'] = (
            f'Best checkpoint not found in {work_dir}. '
            'Expected best_mIoU_iter_*.pth (or best_*_iter_*.pth).')
        return result

    result['best_checkpoint'] = best_ckpt

    cfg = Config.fromfile(config_path)
    # cfg.log_level = 'ERROR'
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = work_dir
    cfg.load_from = best_ckpt

    if args.tta:
        trigger_tta(cfg)

    if args.out is not None:
        exp_out_dir = osp.join(args.out, experiment_name)
        cfg.test_evaluator['output_dir'] = exp_out_dir
        cfg.test_evaluator['keep_results'] = True

    before_dirs = set(list_timestamp_dirs(work_dir))

    runner = Runner.from_cfg(cfg)

    start = time.perf_counter()
    metrics_obj = runner.test()
    elapsed = time.perf_counter() - start

    after_dirs = set(list_timestamp_dirs(work_dir))
    new_dirs = sorted(after_dirs - before_dirs)

    if new_dirs:
        test_dir = new_dirs[-1]
    else:
        test_dir = latest_timestamp_dir(work_dir)

    metrics = normalize_metrics_dict(metrics_obj)
    if not metrics:
        metrics = extract_metrics_from_latest_dir(test_dir)

    classwise_metrics = extract_classwise_metrics_from_log(test_dir)
    for key, value in classwise_metrics.items():
        metrics.setdefault(key, value)

    metrics['time'] = elapsed

    metrics_dir = osp.join(work_dir, args.metrics_subdir)
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_idx = next_metrics_index(metrics_dir)
    metrics_filename = f'{metrics_idx:04d}_metrics.json'
    metrics_path = osp.join(metrics_dir, metrics_filename)

    metrics_payload = {
        'experiment_name': experiment_name,
        'config': config_path,
        'work_dir': work_dir,
        'best_checkpoint': best_ckpt,
        'test_run_dir': test_dir,
        'created_at': dt.datetime.now().isoformat(timespec='seconds'),
        'metrics': metrics,
    }
    write_json(metrics_path, metrics_payload)

    result.update({
        'status': 'success',
        'test_run_dir': test_dir,
        'metrics_file': metrics_path,
        'metrics': metrics,
    })
    return result


def unique_experiment_key(name: str, counts: Dict[str, int]) -> str:
    if name not in counts:
        counts[name] = 1
        return name

    counts[name] += 1
    return f'{name}_{counts[name]}'


def main() -> None:
    args = parse_args()
    schedule_file = osp.abspath(args.schedule_file)

    entries = parse_schedule_file(schedule_file)
    if not entries:
        raise RuntimeError(
            f'No active training entries found in schedule file: {schedule_file}')

    print(f'Loaded {len(entries)} experiment(s) from {schedule_file}')

    all_results: List[Dict[str, Any]] = []

    for idx, (config_path, work_dir) in enumerate(entries, start=1):
        exp_name = experiment_name_from_work_dir(work_dir)
        print(f'\n[{idx}/{len(entries)}] Testing {exp_name}')
        print(f'  config: {config_path}')
        print(f'  work_dir: {work_dir}')

        try:
            result = run_single_experiment(config_path, work_dir, args)
        except Exception as exc:
            result = {
                'experiment_name': exp_name,
                'config': config_path,
                'work_dir': work_dir,
                'status': 'failed',
                'error': str(exc),
            }

        all_results.append(result)

        if result['status'] == 'success':
            print_log(
                f"Saved metrics for {exp_name}: {result['metrics_file']}",
                logger='current')
        else:
            print_log(
                f"Failed {exp_name}: {result.get('error', 'Unknown error')}",
                logger='current',
                level=logging.WARNING)
            if args.strict:
                raise RuntimeError(result.get('error', 'Experiment failed'))

    experiments_map: Dict[str, Dict[str, Any]] = {}
    key_counts: Dict[str, int] = {}
    for result in all_results:
        key = unique_experiment_key(result['experiment_name'], key_counts)
        experiments_map[key] = result

    summary_payload = {
        'generated_at': dt.datetime.now().isoformat(timespec='seconds'),
        'schedule_file': schedule_file,
        'launcher': args.launcher,
        'metrics_subdir': args.metrics_subdir,
        'total_experiments': len(all_results),
        'successful_experiments': sum(
            1 for item in all_results if item.get('status') == 'success'),
        'failed_experiments': sum(
            1 for item in all_results if item.get('status') != 'success'),
        'experiments': experiments_map,
    }

    summary_output = osp.abspath(args.summary_output)
    write_json(summary_output, summary_payload)

    print_table(all_results)

    print('\nConsolidated summary written to:')
    print(summary_output)


if __name__ == '__main__':
    main()
