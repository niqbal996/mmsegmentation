# Copyright (c) OpenMMLab. All rights reserved.
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.fileio import load

def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        # take it as a file path
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            # pkl, json or yaml
            class_weight = load(class_weight)

    return class_weight


def reduce_loss(loss, reduction) -> torch.Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss,
                       weight=None,
                       reduction='mean',
                       avg_factor=None) -> torch.Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

class DetectSPBoundary(nn.Module):
    """
    detect boundary for superpixel, give the superpixel bool mask, return the bool boundary of the superpixel
    """

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, neighbor=8, padding_mode='zeros'):
        """
        padding_mode: 'zeros', 'reflect', 'replicate', 'circular'
        """
        super(DetectSPBoundary, self).__init__()
        # have not been explored
        if kernel_size != 3:
            raise NotImplementedError
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=1, padding=int(kernel_size / 2), bias=False, padding_mode=padding_mode)
        if neighbor == 8:
            a = torch.tensor([[[[-1., -1., -1.],
                                [-1., 8., -1.],
                                [-1., -1., -1.]]]])
        elif neighbor == 4:
            a = torch.tensor([[[[0., -1., 0.],
                                [-1., 4., -1.],
                                [0., -1., 0.]]]])
        else:
            raise NotImplementedError
        # a = a.repeat([1, in_channels, 1, 1])
        a = nn.Parameter(a)
        self.conv.weight = a
        self.conv.requires_grad_(False)

    def forward(self, mask):
        """
        mask:
            (h, w) bool, detect the boundary of the true region
            (b, h, w) long, detect the semantic boundary
        """
        if len(mask.size()) == 2:
            x = mask.float()
            x = x.unsqueeze(dim=0).unsqueeze(dim=0)
            out = self.conv(x)
            out = out.long()
            out = out.squeeze(dim=0).squeeze(dim=0)
            pre_boundary = (out != 0)
            boundary = pre_boundary & mask
            # (h, w)
            return boundary
        elif len(mask.size()) == 3:
            x = mask.float()
            x = x.unsqueeze(dim=1)
            out = self.conv(x)
            out = out.long()
            out = out.squeeze(dim=1)
            pre_boundary = (out != 0)
            # (b, h, w)
            return pre_boundary


class LocalDiscrepancy(nn.Module):

    def __init__(self, in_channels=1, padding_mode='replicate', neighbor=8, l_type="l1"):
        """
        depth-wise conv to calculate the mean of neighbor
        """
        super(LocalDiscrepancy, self).__init__()
        self.type = l_type
        # Force in_channels to 1 because we process single-channel masks,
        # even if they come in a batch. The 'in_channels' from the config
        # might be misleading (e.g., set to num_classes).
        conv_in_channels = 1
        self.mean_conv = nn.Conv2d(in_channels=conv_in_channels, out_channels=conv_in_channels, kernel_size=3,
                                   stride=1, padding=int(3 / 2), bias=False, padding_mode=padding_mode,
                                   groups=conv_in_channels)
        if neighbor == 8:
            a = torch.tensor([[[[1., 1., 1.],
                                [1., 1., 1.],
                                [1., 1., 1.]]]]) / 9
        elif neighbor == 4:
            a = torch.tensor([[[[0., 1., 0.],
                                [1., 1., 1.],
                                [0., 1., 0.]]]]) / 5
        else:
            raise NotImplementedError
        a = a.repeat([conv_in_channels, 1, 1, 1])
        a = nn.Parameter(a)
        self.mean_conv.weight = a
        self.mean_conv.requires_grad_(False)

    def forward(self, x):
        # The input x is expected to be logits, of shape [B, 1, H, W]
        # We apply sigmoid to get probabilities for each mask in the batch.
        p = torch.sigmoid(x)
        mean = self.mean_conv(p)
        l = None
        if self.type == "l1":
            l = torch.abs(p - mean).sum(dim=1)
        elif self.type == "kl":
            # Note: KL divergence for binary case is different.
            # This implementation might be incorrect for sigmoid outputs.
            # Sticking to L1 for now as it's safer.
            l = torch.sum(p * torch.log(p / (mean + 1e-6) + 1e-6), dim=1)
        else:
            raise NotImplementedError("not implemented local soft loss: {}".format(self.type))
        return l