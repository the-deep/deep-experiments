from torch import nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.75):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):

        BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss
        return torch.mean(F_loss)