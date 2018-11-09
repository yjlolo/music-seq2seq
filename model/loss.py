import torch
import torch.nn.functional as F


def mse_loss(output, target, reduction='elementwise_mean'):
    return F.mse_loss(output, target, reduction=reduction)


def pairwise_distances(x):
    t = torch.cat([torch.tensor(F.pairwise_distance(i, x, keepdim=True)) for i in x], dim=1)
    return t


class MSE_loss():
    def __init__(self, reduction='elementwise_mean'):
        self.reduction = reduction

    def __call__(self, output, target):
        return F.mse_loss(output, target, reduction=self.reduction)
