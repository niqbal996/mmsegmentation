import os
dataset_type = 'PhenobenchDatasetAL'
data_root = '/netscratch/naeem/weedsgalore-dataset/cityscapes_binary_labels'

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
    # dict(type='PhenoBenchReduceClasses'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='PhenoBenchReduceClasses'),
    # Keep native label/image size during evaluation to avoid shape mismatch
    # between model predictions (often at original size) and GT masks.
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
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='labels/val'),
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
    dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_label_ids=[1]),
    dict(
        type='InstanceDetectionMetric',
        overlap_thr=0.5,
        overlap_mode='gt',
        crop_label=1,
        weed_label=2,
        ignore_gt_labels_for_metrics=[1],
        ignore_gt_labels_for_precision_fp=[1],
        instance_map_path='/netscratch/naeem/weedsgalore-dataset/cityscapes_binary_labels/plant_instances/val',
        instance_map_suffix='.png',
        vis_output_dir='/netscratch/naeem/mmseg_output/eccv_results/weedsgalore/val',
        vis_area_bins=['0_100'],
        vis_class='weed',
        show_pred_for_detected_only=False,
        # pred_weed_morph_kernel_size=3,   # or 5
        # pred_weed_morph_op='dilation',
        vis_save_weed_island=False,
        vis_save_fp_cases=False,
        # vis_fp_case_max=20,
        vis_fp_show_labels=False,
        vis_per_eval_subdir=False)
]
test_evaluator = val_evaluator