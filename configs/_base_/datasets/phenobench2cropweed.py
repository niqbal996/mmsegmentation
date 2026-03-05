dataset_type_train = 'PhenobenchDataset'
data_root_train = '/netscratch/naeem/phenobench'
dataset_type_val = 'PhenobenchDataset'
data_root_val = '/ds/images/cropandweed/cityscapes_phenobench_format'
# Define your dataset's classes and palette
dataset_meta = dict(
    classes=('Soil', 'Sugarbeet', 'Weeds'),
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
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(
            img_path='train/images',
            seg_map_path='train/semantics'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type_val,
        data_root=data_root_val,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        # indices=list(range(0, 100)),  # Use a subset of val images for quick eval
        data_prefix=dict(
            img_path='leftImg8bit',
            seg_map_path='gtFine'),
        pipeline=test_pipeline))


test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', 
                     iou_metrics=['mIoU'],
                    #  ignore_index=1,
                     )
test_evaluator = val_evaluator