import torch.nn.functional as F


def mse_loss(output, target, reduction='elementwise_mean'):
    return F.mse_loss(output, target, reduction=reduction)
