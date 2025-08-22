# dataset settings
import os
dataset_type_train = 'SyclopsDataset'
dataset_type_val = 'PhenobenchDataset'

data_root_train = '/mnt/e/datasets/sugarbeet_syn_v6'
data_root_val = '/mnt/e/datasets/phenobench'
# /home/niqbal/anaconda3/envs/mmseg_310/lib/python3.10/site-packages/mmcv/transforms/loading.py
# Define your dataset's classes and palette
dataset_meta = dict(
    classes=('background', 'crop', 'weed'),
    palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]],
    subset_ratio=float(os.environ.get('SUBSET_RATIO', 0.1)),
    seed=42,
)

source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhenoBenchReduceClasses'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PhenoBenchReduceClasses'),
    dict(type='PackSegInputs')
]

source_train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type_train,
        data_root=data_root_train,
        metainfo=dataset_meta,
        # subset_ratio=dataset_meta['subset_ratio'],
        # seed=dataset_meta['seed'],
        data_prefix=dict(
            img_path='main_camera/rect',
            seg_map_path='main_camera_annotations/semantics'),
        pipeline=source_train_pipeline))

# TODO Target has to also actively sample. For now, this is just fully supervised. No active part. 
target_train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type_val,
        data_root=data_root_val,
        metainfo=dataset_meta,
        # subset_ratio=dataset_meta['subset_ratio'],
        # seed=dataset_meta['seed'],
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/semantics'),
        pipeline=target_train_pipeline))

train_dataloader = dict(
    dataloader_source=source_train_dataloader,
    dataloader_target=target_train_dataloader,
    # Optionally add more dataloaders here
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_val,
        data_root=data_root_val,
        metainfo=dataset_meta,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/semantics'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator