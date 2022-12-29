from torch import nn
import torch.nn.functional as F
import torch
from typing import Optional


class FocalLoss(nn.Module):
    # EPSILON is used to prevent infinity if some tag proportions are zero
    # valued. See in the constructor
    EPSILON = 1e-10

    def __init__(
        self,
        tag_token_proportions: torch.Tensor,
        device: str,
        gamma: float = 2,
        proportions_pow: float = 1,
    ):
        """
        tag_token_proportions: Contains proportions of positive tokens for each tag.
            Its shape is 1 x num_tags
        """
        super(FocalLoss, self).__init__()

        weight = (
            torch.pow(
                1 / ((tag_token_proportions + FocalLoss.EPSILON) * 2), proportions_pow
            )
            if tag_token_proportions is not None
            else None
        )

        self.gamma = gamma
        self.weight = weight.to(device=device)

    def forward(self, inp, target):
        ce_loss = F.binary_cross_entropy_with_logits(
            inp,
            # generally target is binary, so to be on the safe side convert to float64
            target.to(torch.float64),
            reduction="none",
        )
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma * ce_loss * self.weight).mean()
        return focal_loss
