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


class Emo_loss():
    def __init__(self, effect_epoch, metric='euclidean'):
        self.effect_epoch = effect_epoch
        self.metric = metric

    def __call__(self, output, centroids, epoch):
        if epoch >= self.effect_epoch:
            if self.metric == 'euclidean':
                dist_centroids = pairwise_distances(centroids)
                dist_centroids /= dist_centroids.max()
                dist_centroids.requires_grad_(False)
                dist_centroids.to(output.device)

                dist_output = pairwise_distances(output)
                dist_output = dist_output / dist_output.max()

                diff = torch.sum((dist_output - dist_centroids)**2)

                return diff
            else:
                raise NotImplementedError
        else:
            return torch.zeros(1).to(output.device)
