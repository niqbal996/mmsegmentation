import torch
import torch.nn as nn
from mmseg.registry import MODELS
from .utils import DetectSPBoundary, LocalDiscrepancy

@MODELS.register_module()
class LocalConsistentLoss(nn.Module):
    def __init__(self, in_channels, l_type='l1', loss_weight=1.0):
        super(LocalConsistentLoss, self).__init__()
        self.semantic_boundary = DetectSPBoundary(padding_mode='zeros')
        self.neighbor_dif = LocalDiscrepancy(in_channels=in_channels, padding_mode='replicate', l_type=l_type)
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        assert isinstance(pred, torch.Tensor), 'Expected input to' \
            f'be Tensor, but got {pred.__class__.__name__} instead'
        assert isinstance(target, torch.Tensor), 'Expected target to' \
            f'be Tensor, but got {target.__class__.__name__} instead'
        # assert pred.squeeze(0).shape == target.shape, 'Input and target ' \
        #     'must have same shape,' \
        #     f'but got shapes 1x{pred.shape} and {target.shape}'
        discrepancy = self.neighbor_dif(pred)
        mask = self.semantic_boundary(target)
        mask = mask & (target != 255)
        loss = discrepancy[mask].mean() * self.loss_weight
        return loss 