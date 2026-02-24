import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from monai.networks import one_hot
import numpy as np

class SCNPCEDice(nn.Module):
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
        super(SCNPCEDice, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.rf = receptive_field
        self.weights = 1
        self.ce_loss_fn = _binary_crossentropy_loss
        self.dice_loss_fn = _dice_loss

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        target = torch.unsqueeze(target, 1)

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
        t3 = torch.sigmoid(t1*target + t2*(1-target))

        dice_loss_vc = self.dice_loss_fn(t3, target, self.weights)
        ce_loss_vc = self.ce_loss_fn(t3, target, self.weights)
        return ce_loss_vc + dice_loss_vc

def _dice_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    axis = list([i for i in range(2, len(target.shape))])
    num = 2 * torch.sum(input * target, axis=axis) + 1e-5
    denom = torch.clip( torch.sum(target, axis=axis) + torch.sum(input, axis=axis) + 1e-5, 1e-8)
    dice_loss = (num / denom) * weights
    return -torch.mean(dice_loss)

def _binary_crossentropy_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    axis = list(range(len(input.shape)))
    axis.remove(1)
    ce = -torch.mean( target * torch.log(input + 1e-15) + ((1-target)*(torch.log((1-input) + 1e-15))), axis=axis)
    ce_loss = torch.sum(ce *  weights)
    return ce_loss

