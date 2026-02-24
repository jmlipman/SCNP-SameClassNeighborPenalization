##########################
#### vvvvv SCNP vvvvv ####
##########################

import torch

# Copy-paste this in your code (yes, you only need this :))
class SCNP(torch.nn.Module):
    def __init__(self, dimensions, neighborhood_size=3):
        super(SCNP, self).__init__()

        if dimensions == "2D":
            self.mp = torch.nn.functional.max_pool2d
            self.ns = (neighborhood_size, neighborhood_size)
            self.st = (1, 1)
            self.pad = (neighborhood_size//2, neighborhood_size//2)
        elif dimensions == "3D":
            self.mp = torch.nn.functional.max_pool3d
            self.ns = (neighborhood_size, neighborhood_size, neighborhood_size)
            self.st = (1, 1, 1)
            self.pad = (neighborhood_size//2, neighborhood_size//2)
        else:
            raise Exception("`dimensions` parameters must be either '2D' or '3D'")

    def forward(self, logits, target):
        assert logits.shape == target.shape, "`target` should be one-hot encoded and have the same shape as `logits`"

        # MinPooling in the foreground
        t1 = -self.mp(-(logits*target+9999*(1-target)), self.ns, self.st, self.pad)
        # MaxPooling in the background
        t2 = self.mp((logits*(1-target)-9999*target), self.ns, self.st, self.pad)
        z_tilde = t1*target + t2*(1-target)

        return z_tilde


#############################
#### vvvvv Example vvvvv ####
#############################

# Data
X # torch.Tensor of size (B, C, H, W, (D))

# One-hot encoded labels, with N = number of classes
Y # torch.Tensor of size (B, N, H, W, (D))

# 2D version of SCNP, with a neighborhood size of 3x3
scnp = SCNP("2D", 3)
# 3D version of SCNP, with a neighborhood size of 5x5x5
# scnp = SCNP("3D", 5)

z_logits = model(X)
scnp_logits = scnp(z_logits, Y)

loss = CrossEntropyDiceLoss(scnp_logits, Y)
# loss = CrossEntropyDiceLoss(scnp_logits, Y) + CrossEntropyDiceLoss(z_logits, Y)

loss.backward()...
