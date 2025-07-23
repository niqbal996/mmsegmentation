dataset_type_train = 'PhenobenchDataset'
data_root_train = '/mnt/e/datasets/phenobench/'
dataset_type_val = 'CropAndWeedDataset'
data_root_val = '/mnt/e/datasets/cropandweed_dataset'
# Define your dataset's classes and palette
dataset_meta = dict(
    classes=('background', 'crop', 'weed'),
    palette=[[0, 0, 0], [0, 255, 0], [255, 0, 0]]
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PhenoBenchReduceClasses'),
    dict(type='Resize', scale=(1920, 1088), keep_ratio=False),  # Force exact size
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),  # Crop to multiple of 8
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='CropAndWeed2Phenobench'),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=False),  # Force to multiple of 8
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_train,
        data_root=data_root_train,
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/semantics'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type_val,
        data_root=data_root_val,
        variant='SugarBeet2',
        data_prefix=dict(
            img_path='images',
            seg_map_path='labelIds'),
        pipeline=test_pipeline))


test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator