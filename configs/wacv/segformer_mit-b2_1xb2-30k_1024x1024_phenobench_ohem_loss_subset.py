_base_ = [
    'segformer_mit-b2_1xb2-30k_1024x1024_common.py',
    'datasets/phenobench_subset.py',
]
model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='OhemCrossEntropy',  
            thres=0.7,
            min_kept=100000,
            loss_weight=1.0,
            class_weight=[1.0, 1.0, 5.0]
        ))
)
# train_dataloader = dict(batch_size=1, num_workers=4)
# val_dataloader = dict(batch_size=1, num_workers=4)
# test_dataloader = val_dataloader
