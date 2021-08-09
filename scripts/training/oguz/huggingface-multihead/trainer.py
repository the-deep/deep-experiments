from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from loss import sigmoid_focal_loss, sigmoid_focal_loss_star


class MultiHeadTrainer(Trainer):
    """HuggingFace Trainer compatible with MultiHeadTransformer models.

    Args:
        loss_fn: 'ce', 'focal', 'focal_star'
        loss_weights: weighting applied to different classes
        loss_pos_weights: weighting applied to positive versus negative instances
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        loss_fn: str = "ce",
        loss_weights: Optional[List[float]] = None,
        loss_pos_weights: Optional[List[float]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        if loss_fn == "ce":
            if loss_weights:
                loss_weights = torch.FloatTensor(loss_weights).to('cuda')
            if loss_pos_weights:
                loss_pos_weights = torch.FloatTensor(loss_pos_weights).to('cuda')

            self.loss_fn = torch.nn.BCEWithLogitsLoss(
                weight=loss_weights, pos_weight=loss_pos_weights
            )
        elif loss_fn == "focal":
            assert (
                loss_weights is None and loss_pos_weights is None
            ), "Does not support weighting with focal loss"
            self.loss_fn = sigmoid_focal_loss
        elif loss_fn == "focal_star":
            assert (
                loss_weights is None and loss_pos_weights is None
            ), "Does not support weighting with focal loss-star"
            self.loss_fn = sigmoid_focal_loss_star
        else:
            raise "Unknown loss function"

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        groups = inputs.pop("groups")

        if model.iterative:
            pred_groups, pred_labels = model(inputs, gt_groups=groups)

            # calculate group loss
            loss_group = self.loss_fn(
                pred_groups.view(-1, groups.shape[-1]),
                groups.float().view(-1, groups.shape[-1]),
            )
        else:
            pred_labels = model(inputs)

        # calculate label loss
        loss = self.loss_fn(
            pred_labels.view(-1, labels.shape[-1]),
            labels.float().view(-1, labels.shape[-1]),
        )
        logits = {"logits": pred_labels}

        # update loss + logits for iterative
        if model.iterative:
            loss = loss + loss_group
            logits["logits_group"] = pred_groups

        return (loss, logits) if return_outputs else loss
