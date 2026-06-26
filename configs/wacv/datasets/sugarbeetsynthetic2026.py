# dataset settings
synthetic_dataset_type = 'SugarBeetSynthetic2026Dataset'
real_dataset_type = 'PhenobenchDatasetAL'

synthetic_data_root = '/mnt/e/datasets/sugarbeet_syn_v6/'
real_data_root = '/mnt/e/datasets/phenobench'
# Define your dataset's classes and palette
dataset_meta = dict(
    classes=('background', 'crop', 'weed'),
    palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]],
    subset_ratio=1.0,
    seed=42,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(1.0, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
real_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhenoBenchReduceClasses'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackSegInputs')
]

semantic_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
    ignore_index=255,
    ignore_label_ids=[])

instance_evaluator = dict(
    type='InstanceDetectionMetric',
    object_iog_thr=0.05,
    object_iop_thr=0.05,
    class0_label=0,
    class1_label=1,
    class2_label=2,
    instance_map_path='/mnt/e/datasets/phenobench/val/plant_instances',
    instance_map_suffix='.png',
    instance_map_subdirs=('plant_instances', 'instance_segmentation'),
    # vis_fp_case_max=20,
    # vis_bg_alpha=0.35,
    pred_island_min_area=10,
    class_names=('background/soil', 'crop', 'weed'),
    ignore_index=255,
    ignore_label_ids=[],
    allow_semantic_instance_fallback=True,
    resize_instance_to_gt=True)

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=synthetic_dataset_type,
        data_root=synthetic_data_root,
        metainfo=dataset_meta,
        # subset_ratio=dataset_meta['subset_ratio'],
        # seed=dataset_meta['seed'],
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='main_camera_annotations/semantics/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=synthetic_dataset_type,
        data_root=synthetic_data_root,
        metainfo=dataset_meta,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='main_camera_annotations/semantics/val'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=real_dataset_type,
        data_root=real_data_root,
        subset_ratio=1.0,
        sample_list='/mnt/e/datasets/phenobench/phenobench_train_list.txt',
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/semantics'),
        pipeline=real_test_pipeline))

val_evaluator = [semantic_evaluator]
test_evaluator = [semantic_evaluator, instance_evaluator]
