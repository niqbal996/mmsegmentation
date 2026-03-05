# dataset settings
import os
dataset_type_train = 'SyclopsDatasetDilatedWeedInstances'
dataset_type_val = 'SyclopsDatasetDilatedWeedInstances'
data_root = '/path/to/sugarbeetsynthetic2026/root/'
# Define your dataset's classes and palette
dataset_meta = dict(
    classes=('background', 'crop', 'weed'),
    palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]],
    subset_ratio=float(os.environ.get('SUBSET_RATIO', 1.0)),
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
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'seg_map_path', 'instance_map_path',
                   'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
                   'flip', 'flip_direction', 'reduce_zero_label'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type_train,
        data_root=data_root,
        metainfo=dataset_meta,
        instance_map_path=os.path.join(data_root, 'main_camera_annotations/instance_segmentation'),
        img_suffix='.png',
        seg_map_suffix='.png',
        instance_map_suffix='.npz',
        dilate_kernel_size=10,
        dilate_iterations=3,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='main_camera_annotations/semantics/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_val,
        data_root=data_root,
        metainfo=dataset_meta,
        instance_map_path=os.path.join(data_root, 'main_camera_annotations/instance_segmentation'),
        img_suffix='.png',
        seg_map_suffix='.png',
        instance_map_suffix='.npz',
        dilate_kernel_size=10,
        dilate_iterations=3,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='main_camera_annotations/semantics/val'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(
        type='InstanceDetectionMetric',
        overlap_thr=0.05,
        overlap_mode='gt',
        crop_label=1,
        weed_label=2,
        instance_map_suffix='.npz',
        vis_output_dir='/path/to/mmseg_output/eccv_results/instance_detection_vis',
        vis_area_bins=['100_200'],
        vis_class='weed',
        show_pred_for_detected_only=True)
]
test_evaluator = val_evaluator