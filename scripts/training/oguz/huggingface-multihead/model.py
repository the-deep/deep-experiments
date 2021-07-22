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
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.config.problem_type = "multi_label_classification"
        self.backbone.trainable = not freeze_backbone

        self.dropout = torch.nn.Dropout(dropout)
        self.heads = torch.nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                build_mlp(
                    depth=num_layers,
                    in_features=self.backbone.config.dim,
                    middle_features=np.sqrt(num_classes[i] * self.backbone.config.dim),
                    out_features=num_classes[i],
                    bias=True,
                    batchnorm=False,
                    final_norm=False,
                )
            )

    def forward(self, inputs):
        outputs = self.backbone(**inputs)
        last_hidden_states = torch.mean(outputs.last_hidden_state, axis=1)
        outputs = self.dropout(last_hidden_states)

        outs = []
        for head in self.heads:
            outs.append(head(outputs))
        return torch.cat(outs, axis=-1)
