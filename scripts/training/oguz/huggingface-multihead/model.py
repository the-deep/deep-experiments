from typing import List

import numpy as np
import torch
from transformers import PreTrainedModel

from utils import build_mlp


class MultiHeadTransformer(torch.nn.Module):
    def __init__(
        self,
        backbone: PreTrainedModel,
        num_heads: int,
        num_classes: List[int],
        num_layers: int = 1,
        dropout: float = 0.3,
        pooling: bool = False,
        freeze_backbone: bool = False,
        iterative: bool = False,
        use_gt_training: bool = True,
    ):
        super().__init__()
        self.pooling = pooling
        self.iterative = iterative
        self.use_gt_training = use_gt_training

        self.backbone = backbone
        self.backbone.config.problem_type = "multi_label_classification"
        self.backbone.trainable = not freeze_backbone

        self.dropout = torch.nn.Dropout(dropout)
        self.heads = torch.nn.ModuleList()

        mlp_params = {
            "depth": num_layers,
            "in_features": self.backbone.config.dim,
            "bias": True,
            "batchnorm": False,
            "final_norm": False,
        }

        if iterative:
            build_mlp(
                middle_features=np.sqrt(len(num_classes) * self.backbone.config.dim),
                out_features=len(num_classes),
                **mlp_params
            )

        for i in range(num_heads):
            self.heads.append(
                build_mlp(
                    middle_features=np.sqrt(num_classes[i] * self.backbone.config.dim),
                    out_features=num_classes[i],
                    **mlp_params
                )
            )

    def forward(self, inputs, gt_groups=None):
        # get hidden representation
        backbone_outputs = self.backbone(**inputs)
        if self.pooling:
            last_hidden_states = torch.mean(backbone_outputs.last_hidden_state, axis=1)
        else:
            last_hidden_states = backbone_outputs.last_hidden_state[:, 0, :]
        hidden = self.dropout(last_hidden_states)

        if self.iterative:
            # execute super-classification task
            out_groups = self.heads[0](hidden)

            # get sample groups
            # TODO: dynamic threshold?
            groups = gt_groups if self.training and self.use_gt_training else out_groups > 0.5

            # execute each classification task
            out_targets = []
            for i, head in enumerate(self.heads[1:]):
                out_target = head(hidden)
                out_targets.append(
                    torch.where(
                        torch.repeat(groups[:, i], out_target.shape[1]),
                        out_target,
                        torch.zeros_like(out_target),
                    )
                )
            out_targets = torch.cat(out_targets, axis=-1)
        else:
            # execute each classification task
            out_targets = []
            for head in self.heads:
                out_targets.append(head(hidden))
            out_targets = torch.cat(out_targets, axis=-1)

        return (out_groups, out_targets) if self.iterative else out_targets
