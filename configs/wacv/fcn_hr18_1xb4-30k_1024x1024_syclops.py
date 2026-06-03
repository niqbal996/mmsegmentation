_base_ = [
    '../_base_/models/fcn_hr18.py', 
    '../_base_/datasets/syclops.py',
    '../_base_/default_runtime.py', 
]
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
num_classes = 3
iters = 30000
interval = 1000
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=num_classes),
   )
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]
# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=interval,
        max_keep_ckpts=1,
        # Track best checkpoint by class-wise IoU for weeds.
        save_best='IoU_weed',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))