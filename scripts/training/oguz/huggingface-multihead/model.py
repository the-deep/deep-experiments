from typing import List, Optional

import numpy as np
import torch
from transformers import PreTrainedModel

from utils import build_mlp

ZERO_SIGMOID_INVERSE = -10


class MultiHeadTransformer(torch.nn.Module):
    """Multi-task MLP classifier using the same transformer backbone.

    Args:
        backbone: Pre-trained transformer.
        num_heads: Number of classification tasks.
        num_classes: List of number of classes in each task.
        num_layers: Depth of MLP classfier heads.
        dropout: Rate of dropout in tranformer output before MLP classifiers.
        pooling: If true, classifiers use averaged representations of all symbols.
                 If false, classifiers use representation of the start symbol.
        freeze_backbone: Only train classifiers with backbone.
        iterative: Adds an additional classification head for coarser _group_ task.
            Only relevant if the task involves (coarse, fine-grained) labels.
            If enabled, an additional classifier is first used to predict the coarse
            label and the other heads predict the coarse label. The coarse classifier
            acts as a filter, i.e., if a negative prediction occurs for a coarse label,
            all predictions for labels in that group are set to high negative values.
        use_gt_training: uses ground truth group values in the training
            Only relevant if iterative is set to True.
        backbone_dim: dimension of the backbone transformer
            Set if the dimension is not accessible through the config of backbone.
    """

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
        backbone_dim: Optional[int] = None,
    ):
        super().__init__()
        self.pooling = pooling
        self.iterative = iterative
        self.use_gt_training = use_gt_training

        self.backbone = backbone
        self.backbone.config.problem_type = "multi_label_classification"
        self.backbone.trainable = not freeze_backbone

        if not hasattr(self.backbone.config, "dim"):
            assert backbone_dim is not None, "Model config does not include output dim!"
            dim = backbone_dim
        else:
            dim = self.backbone.config.dim

        self.dropout = torch.nn.Dropout(dropout)
        self.heads = torch.nn.ModuleList()

        mlp_params = {
            "depth": num_layers,
            "in_features": dim,
            "bias": True,
            "batchnorm": False,
            "final_norm": False,
        }

        if iterative:
            self.heads.append(
                build_mlp(
                    middle_features=np.floor(np.sqrt(len(num_classes) * dim)).astype(int),
                    out_features=len(num_classes),
                    **mlp_params
                )
            )

        for i in range(num_heads):
            self.heads.append(
                build_mlp(
                    middle_features=np.floor(np.sqrt(num_classes[i] * dim)).astype(int),
                    out_features=num_classes[i],
                    **mlp_params
                )
            )

    def forward(self, inputs, gt_groups=None, group_threshold=0.5):
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

            # get group predictions
            # TODO: dynamic threshold (per group?)
            groups = (
                gt_groups
                if self.training and self.use_gt_training
                else out_groups > group_threshold
            )

            # execute each classification task
            out_targets = []
            for i, head in enumerate(self.heads[1:]):
                out_target = head(hidden)
                out_targets.append(
                    torch.where(
                        torch.repeat_interleave(groups[:, i : i + 1], out_target.shape[1], dim=1),
                        out_target,  # classifer output if group is predicted as `positive`
                        torch.zeros_like(out_target)
                        + ZERO_SIGMOID_INVERSE,  # zero if group is predicted as `negative`
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
