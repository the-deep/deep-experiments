from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alphas, gamma=0.2):
        super(FocalLoss, self).__init__()
        self.alphas = alphas
        self.gamma = gamma

    def forward(self, outputs, targets, weighted: bool):
        if weighted:
            BCE_loss = F.binary_cross_entropy_with_logits(
                outputs, targets, reduction="mean", pos_weight=self.alphas
            )
        else:
            BCE_loss = F.binary_cross_entropy_with_logits(
                outputs, targets, reduction="mean"
            )
        """
        #TODO: finish implementing and testing focal loss
        pt = torch.exp(-BCE_loss)
        row_loss = ((1 - pt) ** self.gamma) * BCE_loss
        row_mean = torch.mean(row_loss, 0)

        F_loss = torch.dot(row_mean, self.alphas)"""

        return BCE_loss
