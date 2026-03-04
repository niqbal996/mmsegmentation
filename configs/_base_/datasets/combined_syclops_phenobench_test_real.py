# dataset settings
import os

synthetic_dataset_type = 'SyclopsDataset'
real_dataset_type = 'PhenobenchDataset'

data_root_synthetic = '/netscratch/naeem/sugarbeet_syn_v6'
data_root_real = '/netscratch/naeem/phenobench'

batch_size = int(os.environ.get('TRAIN_BATCH_SIZE', 4))
real_subset_ratio = float(os.environ.get('REAL_SUBSET_RATIO', 0.05))
print('Using ============ > REAL_SUBSET_RATIO:', real_subset_ratio)
# Source ratio per training batch: [synthetic, real]
data_ratio = [0.5, 0.5]

dataset_meta = dict(
    classes=('background', 'crop', 'weed'),
    palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]],
)

real_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhenoBenchReduceClasses'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
syn_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
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

synthetic_train_dataset = dict(
    type=synthetic_dataset_type,
    data_root=data_root_synthetic,
    metainfo=dataset_meta,
    data_prefix=dict(
        img_path='images/train',
        seg_map_path='main_camera_annotations/semantics/train'),
    pipeline=syn_train_pipeline)

real_train_dataset = dict(
    type=real_dataset_type,
    data_root=data_root_real,
    metainfo=dataset_meta,
    data_prefix=dict(
        img_path='train/images',
        seg_map_path='train/semantics'),
    pipeline=real_train_pipeline)

# Optional standalone source dataloaders (not used by runner directly).
synthetic_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=synthetic_train_dataset)

real_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=real_train_dataset)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    # Keep shuffle=False so CombinedDataset preserves exact per-batch ratio.
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(
        type='CombinedDataset',
        datasets=[synthetic_train_dataset, real_train_dataset],
        data_ratio=data_ratio,
        batch_size=batch_size,
        subset_ratios=[1.0, real_subset_ratio],
        subset_seed=42))


val_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=real_dataset_type,
        data_root=data_root_real,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/semantics'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
