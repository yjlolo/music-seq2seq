import torch
import torch.nn.functional as F


def pairwise_distances(x):
    t = torch.cat([torch.tensor(F.pairwise_distance(i, x, keepdim=True)) for i in x], dim=1)
    return t


class MSE_loss():
    def __init__(self, effect_epoch, reduction='elementwise_mean'):
        self.effect_epoch = effect_epoch
        self.reduction = reduction

    def __call__(self, output, target, epoch):
        if epoch >= self.effect_epoch:
            return F.mse_loss(output, target, reduction=self.reduction)
        else:
            return torch.zeros(1).to(output.device)


class Emo_loss():
    def __init__(self, effect_epoch, scale, metric='euclidean'):
        self.effect_epoch = effect_epoch
        self.scale = scale
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

                return diff * self.scale
            else:
                raise NotImplementedError
        else:
            return torch.zeros(1).to(output.device)
