import torch
import torch.nn as nn
from mmseg.registry import MODELS

@MODELS.register_module()
class NegativeLearningLoss(nn.Module):
    def __init__(self, threshold=0.05, loss_weight=1.0):
        super(NegativeLearningLoss, self).__init__()
        self.threshold = threshold
        self.loss_weight = loss_weight

    def forward(self, predict):
        mask = (predict < self.threshold).detach()
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, device=predict.device, dtype=predict.dtype)
        negative_loss_item = -1 * mask * torch.log(1 - predict + 1e-6)
        negative_loss = torch.sum(negative_loss_item) / torch.sum(mask)
        negative_loss = negative_loss * self.loss_weight
        return negative_loss