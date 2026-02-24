# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
from monai.networks import one_hot
import torch.nn.functional as F
import numpy as np

class SCNPCEDiceLoss(nn.Module):
    def __init__(self, weight_ce=1, weight_dice=1, ignore_label=None,
                 receptive_field=3):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(SCNPCEDiceLoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.rf = receptive_field
        self.weights = 1
        self.dice_loss_fn = _dice_loss
        self.ce_loss_fn = _crossentropy_loss

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        target = one_hot(target.unsqueeze(1), num_classes=net_output.shape[1])

        if net_output.ndim == 4:
            mp = torch.nn.functional.max_pool2d
            rf = (self.rf, self.rf)
            st = (1, 1)
            pad = (self.rf//2, self.rf//2)
        else:
            mp = torch.nn.functional.max_pool3d
            rf = (self.rf, self.rf, self.rf)
            st = (1, 1, 1)
            pad = (self.rf//2, self.rf//2, self.rf//2)

        t1 = -mp(-(net_output*target+9999*(1-target)), rf, st, pad)
        t2 = mp((net_output*(1-target)-9999*target), rf, st, pad)
        t3 = torch.softmax(t1*target + t2*(1-target), 1)

        ce_loss_vc = self.ce_loss_fn(t3, target, self.weights)
        dice_loss_vc = self.dice_loss_fn(t3, target, self.weights)
        return ce_loss_vc + dice_loss_vc

class DeepLabCE(nn.Module):
    """
    Hard pixel mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Paper: DeeperLab: Single-Shot Image Parser
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33  # noqa
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its
            value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, top_k_percent_pixels=1.0, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def forward(self, logits, labels, weights=None):
        #from IPython import embed; embed(); asd
        if weights is None:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        else:
            # Apply per-pixel loss weights.
            pixel_losses = self.criterion(logits, labels) * weights
            pixel_losses = pixel_losses.contiguous().view(-1)
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()
        #from IPython import embed; embed(); asd
        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()

def _dice_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    axis = list([i for i in range(2, len(target.shape))])
    num = 2 * torch.sum(input * target, axis=axis) + 1e-5
    denom = torch.clip( torch.sum(target, axis=axis) + torch.sum(input, axis=axis) + 1e-5, 1e-8)
    dice_loss = (num / denom) * weights
    return -torch.mean(dice_loss)

def _crossentropy_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    #ce = torch.sum(target * torch.log(input + 1e-15), axis=1)
    #ce_loss = -torch.mean(ce)
    axis = list(range(len(input.shape)))
    axis.remove(1)
    ce = -torch.mean(target * torch.log(input + 1e-15), axis=axis)
    ce_loss = torch.sum(ce *  weights)
    return ce_loss

