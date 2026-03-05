import os
dataset_type = 'PhenobenchDatasetAL'
data_root = '/path/to/phenobench/root/'

# Define your dataset's classes and palette
dataset_meta = dict(
    classes=('background', 'crop', 'weed'),
    palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]],
    subset_ratio=float(os.environ.get('SUBSET_RATIO', 1.0)),
    random_seed=42,
)

train_pipeline = [
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
    dict(type='PhenoBenchReduceClasses'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        subset_ratio=dataset_meta['subset_ratio'],
        # random_seed=dataset_meta['random_seed'],
        sample_list='/netscratch/naeem/phenobench/phenobench_train_list.txt',
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/semantics'),
        pipeline=train_pipeline))
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
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(
#             img_path='test/images',
#             seg_map_path='test/semantics'),
#         pipeline=test_pipeline))

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(
        type='InstanceDetectionMetric',
        overlap_thr=0.01,
        overlap_mode='gt',
        crop_label=1,
        weed_label=2,
        instance_map_path='/path/to/phenobench/root/val/plant_instances',
        instance_map_suffix='.png',
        vis_output_dir='/path/to/mmseg_output/eccv_results/syn2real_0.01',
        vis_area_bins=['100_200'],
        vis_class='weed',
        show_pred_for_detected_only=True)
]
test_evaluator = val_evaluator