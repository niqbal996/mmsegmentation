# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import re
import warnings
from glob import glob

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
register_all_modules()

from mmseg.registry import RUNNERS


ITER_PATTERN = re.compile(r'_iter_(\d+)\.pth$')


def parse_iter_from_ckpt(path):
    match = ITER_PATTERN.search(osp.basename(path))
    if not match:
        return -1
    return int(match.group(1))


def find_best_checkpoint(work_dir):
    primary = glob(osp.join(work_dir, 'best_mIoU_iter_*.pth'))
    fallback = glob(osp.join(work_dir, 'best_*_iter_*.pth'))
    candidates = primary if primary else fallback

    if not candidates:
        return None

    candidates.sort(key=lambda p: (parse_iter_from_ckpt(p), p), reverse=True)
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-after-training',
        action='store_true',
        help='evaluate the checkpoint after training')
    parser.add_argument(
        '--log-level',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        help='override config log level')
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='suppress most mmengine logs and Python warnings')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.quiet:
        warnings.filterwarnings('ignore')
        cfg.log_level = 'ERROR'
    elif args.log_level is not None:
        cfg.log_level = args.log_level

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()

    # test the best checkpoint after training
    if args.eval_after_training:
        # Match generate_metrics.py behavior: prefer mIoU checkpoint and
        # fallback to any best_* checkpoint saved by CheckpointHook.
        best_ckpt_path = find_best_checkpoint(runner.work_dir)

        if best_ckpt_path and osp.exists(best_ckpt_path):
            runner.load_checkpoint(best_ckpt_path)
            runner.test()
        else:
            print_log(
                'Best checkpoint is not found, please check your validation '
                'configuration.',
                logger='current',
                level=logging.WARNING)


if __name__ == '__main__':
    main()
