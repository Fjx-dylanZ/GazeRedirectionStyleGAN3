import torch.nn as nn
import torch
import torch.nn.functional as F


class GazeHeadLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_hat, reduction="mean"):
        assert y.shape == y_hat.shape
        assert y.shape[1] == 2
        loss = gaze_angular_loss(y, y_hat, reduction=reduction)
        return loss


def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, eps=1e-6)
    sim = F.hardtanh(sim, -1.0, 1.0)
    return torch.acos(sim) * (180 / torch.pi)


def pitchyaw_to_vector(pitchyaws):
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)


def gaze_angular_loss(y, y_hat, reduction='mean'):
    y = pitchyaw_to_vector(y)
    y_hat = pitchyaw_to_vector(y_hat)
    loss = nn_angular_distance(y, y_hat)
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        print("assuming reduction is None")
    return loss