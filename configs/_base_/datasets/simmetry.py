dataset_type = 'SimmetryDataset'
# data_root = '/mnt/e/projects/TFP/2025-09-02_Testdatensatz/cityscapes'
# data_root = '/mnt/e/projects/TFP/simmetry_converted_dataset/cityscapes'
# data_root = '/netscratch/naeem/tfp_project/simmetry_dataset_mini/cityscapes'
data_root = '/netscratch/naeem/tfp_project/amazone_p4ai_test_dataset/cityscapes'

# Define your dataset's classes and palette
# dataset_meta = dict(
#     classes=('Ground', 'Onions', 'Gaensefuss', 'Hirse', 'Labkraut'),
#     palette=[
#         [0, 0, 0], 
#         [0, 255, 0], 
#         [255, 0, 0], 
#         [255, 255, 0], 
#         [0, 0, 255]]
# )
# weeds 3 classes
# dataset_meta = dict(
#     classes=('Ground', 'Gaensefuss', 'Hirse', 'Labkraut'),
#     palette=[
#         [0, 0, 0], 
#         [0, 255, 0], 
#         [255, 0, 0], 
#         [255, 255, 0]]
# )

# weeds 1 class
# dataset_meta = dict(
#     classes=('Ground', 'Weeds'),
#     palette=[
#         [0, 0, 0], 
#         [0, 255, 0]]
# )

# # phenobench format
dataset_meta = dict(
    classes=('Ground', 'Weeds', 'Onions'),
    palette=[
        [0, 0, 0], 
        [0, 255, 0], 
        [255, 0, 0]]
)

# # onions only
# dataset_meta = dict(
#     classes=('Ground', 'Onions'),
#     palette=[
#         [0, 0, 0], 
#         [0, 255, 0]]
# )

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', zero_indexed=False),
    # TODO change size
    # dict(type='TFP_Onionsonly'),
    # dict(type='TFP_Phenobench_3_class'),
    # dict(type='TFP_Weeds_only_1_class'),
    # dict(type='TFP_Weeds_only_3_class'),
    dict(type='RandomResize', scale=(512*2, 512*4), ratio_range=(0.7, 1.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048), keep_ratio=True),
    # dict(type='TFP_Onionsonly'),
    # dict(type='TFP_Phenobench_3_class'),
    # dict(type='TFP_Weeds_only_1_class'),
    # dict(type='TFP_Weeds_only_3_class'),
    dict(type='LoadAnnotations', zero_indexed=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator