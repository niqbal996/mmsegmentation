_base_ = [
    'segformer_mit-b2_1xb2-30k_1024x1024_common.py',
    'datasets/phenobench_subset.py',
]
# train_dataloader = dict(batch_size=1, num_workers=4)
# val_dataloader = dict(batch_size=1, num_workers=4)
# test_dataloader = val_dataloader
