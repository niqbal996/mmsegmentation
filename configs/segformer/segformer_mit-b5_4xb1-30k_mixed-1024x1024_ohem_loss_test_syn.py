_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/combined_syclops_phenobench_test_syn.py',
    '../_base_/default_runtime.py',
]
crop_size = (1024, 1024)
num_iters = 30000
num_classes = 3
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=num_classes,
        loss_decode=dict(
            type='OhemCrossEntropy',  
            thres=0.7,
            min_kept=100000,
            loss_weight=1.0,
            class_weight=[1.0, 1.0, 5.0]
            ),
        )
    )

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=num_iters,
        by_epoch=False,
    )
]

# train_dataloader = dict(batch_size=1, num_workers=4)
# val_dataloader = dict(batch_size=1, num_workers=4)
# test_dataloader = val_dataloader
