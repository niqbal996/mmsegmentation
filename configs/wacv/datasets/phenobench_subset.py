# dataset settings
import os

dataset_type = 'PhenobenchDataset'
data_root = '/mnt/e/datasets/phenobench'

batch_size = int(os.environ.get('TRAIN_BATCH_SIZE', 4))
real_subset_ratio = float(os.environ.get('REAL_SUBSET_RATIO', 0.1))
print('Using ============ > REAL_SUBSET_RATIO:', real_subset_ratio)

dataset_meta = dict(
    classes=('background', 'crop', 'weed'),
    palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]],
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhenoBenchReduceClasses'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhenoBenchReduceClasses'),
    dict(type='PackSegInputs')
]

real_train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=dataset_meta,
    data_prefix=dict(
        img_path='train/images',
        seg_map_path='train/semantics'),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(
        type='CombinedDataset',
        datasets=[real_train_dataset],
        data_ratio=[1.0],
        batch_size=batch_size,
        subset_ratios=[real_subset_ratio],
        subset_seed=42))

val_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/semantics'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
