import torch
from nnunetv2.training.loss.dice import SoftDiceLoss
from torch import nn
# If you don't want to install monai, you can copy-paste this function
# https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/utils.py#L170
from monai.networks import one_hot


def _dice_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    axis = list([i for i in range(2, len(target.shape))])
    num = 2 * torch.sum(input * target, axis=axis) + 1e-5
    denom = torch.clip( torch.sum(target, axis=axis) + torch.sum(input, axis=axis) + 1e-5, 1e-8)
    dice_loss = (num / denom) * weights
    return -torch.mean(dice_loss)

def _crossentropy_loss(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    axis = list(range(len(input.shape)))
    axis.remove(1)
    ce = -torch.mean(target * torch.log(input + 1e-15), axis=axis)
    ce_loss = torch.sum(ce *  weights)
    return ce_loss

class SCNPCEDice(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss, receptive_field=3):

        super(SCNPCEDice, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.rf = receptive_field
        self.weights = 1
        self.dice_loss_fn = _dice_loss
        self.ce_loss_fn = _crossentropy_loss

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        target = one_hot(target, num_classes=net_output.shape[1])

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

