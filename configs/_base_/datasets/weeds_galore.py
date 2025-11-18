dataset_type = 'WeedsGalore'
data_root = '/mnt/e/datasets/weedsgalore-dataset/cityscapes_binary_labels'

# # phenobench format
dataset_meta = dict(
    classes=('Ground', 'Crops', 'Weeds'),
    palette=[
        [0, 0, 0], 
        [0, 255, 0], 
        [255, 0, 0]]
)
# dataset_meta = dict(
#     classes=('Ground', 'Crops', 'Weeds', 'Weed_1', 'Weed_2', 'Weed_3'),
#     palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 0, 255], [255, 0, 255]]
# )

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', zero_indexed=False),
    dict(type='RandomResize', scale=(512*2, 512*3), ratio_range=(0.7, 1.0), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=(600, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', zero_indexed=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ignore_index=1,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='labels/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ignore_index=1,
        # reduce_zero_label=True,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='labels/train'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ignore_index=1,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='labels/train'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', 
                     iou_metrics=['mIoU'],
                     ignore_index=1,
                     )
test_evaluator = val_evaluator