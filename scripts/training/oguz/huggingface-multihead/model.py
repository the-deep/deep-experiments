from typing import List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedModel

from utils import build_mlp

ZERO_SIGMOID_INVERSE = -10


class MultiTargetTransformer(torch.nn.Module):
    """Multi-target MLP classifier head that is able to handle group structure in
    multi-label classifications problems (e.g., 6 label classification with two groups:
    [A, B, C], [D, E, F]).


    Args:
        num_heads: Number of classification groups.
        num_classes: List of number of classes in each group.
        num_layers: Depth of MLP classfier heads.
        dropout: Rate of dropout in tranformer output before MLP classifiers.
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
        num_classes: List[int] = [1],
        num_layers: int = 1,
        iterative: bool = False,
        use_gt_training: bool = True,
        backbone_dim: Optional[int] = None,
    ):
        super().__init__()
        self.use_gt_training = use_gt_training
        self.heads = torch.nn.ModuleList()

        mlp_params = {
            "depth": num_layers,
            "in_features": backbone_dim,
            "bias": True,
            "batchnorm": False,
            "final_norm": False,
        }

        if iterative:
            self.heads.append(
                build_mlp(
                    middle_features=np.floor(np.sqrt(len(num_classes) * backbone_dim)).astype(int),
                    out_features=len(num_classes),
                    **mlp_params
                )
            )

        for num_cls in num_classes:
            self.heads.append(
                build_mlp(
                    middle_features=np.floor(np.sqrt(num_cls * backbone_dim)).astype(int),
                    out_features=num_cls,
                    **mlp_params
                )
            )

    def forward(self, inputs, gt_groups=None, group_threshold=0.5):
        if self.iterative:
            # execute super-classification task
            out_groups = self.heads[0](inputs)

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
                out_target = head(inputs)
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
                out_targets.append(head(inputs))
            out_targets = torch.cat(out_targets, axis=-1)

        return (out_groups, out_targets) if self.iterative else out_targets


class MultiHeadTransformer(torch.nn.Module):
    """Multi-task classifier each supporting multi-target groups (MultiTargetTransformer)
    using the same transformer backbone.

    Args:
        backbone: Pre-trained transformer.
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
        num_classes: Optional[Union[List[int], List[List[int]]]],
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

        if isinstance(num_classes[0], int):
            num_classes = [num_classes]

        self.dropout = torch.nn.Dropout(dropout)
        self.heads = torch.nn.ModuleList()

        for num_cls in num_classes:
            self.heads.append(
                MultiTargetTransformer(
                    num_classes=num_cls,
                    num_layers=num_layers,
                    iterative=iterative,
                    use_gt_training=use_gt_training,
                    backbone_dim=dim,
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

        # execute forward-pass for all heads
        groups, targets = [], []
        for idx, head in enumerate(self.heads):
            if self.iterative:
                out_groups, out_targets = head(
                    hidden,
                    gt_groups=gt_groups[idx] if isinstance(gt_groups, list) else None,
                    group_threshold=group_threshold[idx]
                    if isinstance(group_threshold, list)
                    else group_threshold,
                )
                groups.append(out_groups)
            else:
                out_targets = head(hidden)
            targets.append(out_targets)

        return (groups, targets) if self.iterative else targets
