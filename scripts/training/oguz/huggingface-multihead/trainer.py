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

"""Trainers here only work with flattened datasets."""


class MultiTargetTrainer(Trainer):
    """HuggingFace Trainer compatible with MultiTargetTransformer models.

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
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
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
                loss_weights = torch.FloatTensor(loss_weights).to("cuda")
            if loss_pos_weights:
                loss_pos_weights = torch.FloatTensor(loss_pos_weights).to("cuda")

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

        if model.iterative:
            groups = inputs.pop("groups")
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


class MultiHeadTrainer(MultiTargetTrainer):
    """HuggingFace Trainer compatible with MultiHeadTransformer models.

    Args:
        loss_fn: 'ce', 'focal', 'focal_star'
        loss_weights: weighting applied to different classes
        loss_pos_weights: weighting applied to positive versus negative instances
        tasks: names of the tasks
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
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        loss_fn: str = "ce",
        loss_weights: Optional[Union[List[float], List[List[float]]]] = None,
        loss_pos_weights: Optional[Union[List[float], List[List[float]]]] = None,
        targets: Optional[Union[str, List[str]]] = "",
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
                loss_weights = [
                    torch.FloatTensor(_loss_weights).to("cuda") for _loss_weights in loss_weights
                ]
            else:
                loss_weights = [None for _ in targets]
            if loss_pos_weights:
                loss_pos_weights = [
                    torch.FloatTensor(_loss_pos_weights).to("cuda")
                    for _loss_pos_weights in loss_pos_weights
                ]
            else:
                loss_pos_weights = [None for _ in targets]

            self.loss_fn = [
                torch.nn.BCEWithLogitsLoss(weight=_loss_weights, pos_weight=_loss_pos_weights)
                for (_loss_weights, _loss_pos_weights) in zip(loss_weights, loss_pos_weights)
            ]
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

        if isinstance(targets, str):
            self.targets = [""]
        else:
            self.targets = targets

    def compute_loss(self, model, inputs, return_outputs=False):
        # collect labels
        labels = []
        for _target in self.targets:
            labels.append(inputs.pop(f"{_target}_labels"))

        # get the model predictions
        if model.iterative:
            # collect group gts
            groups = []
            for target in self.targets:
                groups.append(inputs.pop(f"{target}_groups"))
            pred_groups, pred_labels = model(inputs, gt_groups=groups)
        else:
            pred_labels = model(inputs)

        logits = {}
        loss = torch.tensor(0)
        for idx, _target in enumerate(self.targets):
            # get labels, preds and register logits
            _labels = labels[idx]
            _pred_labels = pred_labels[idx]
            logits[f"{_target}_logits"] = _pred_labels

            # calculate label loss
            _loss = self.loss_fn[idx](
                _pred_labels.view(-1, _labels.shape[-1]),
                _labels.view(-1, _labels.shape[-1]).float(),
            )

            if model.iterative:
                # get groups, preds and register logits
                _groups = inputs.pop(f"{_target}_groups")
                _pred_groups = pred_groups[idx]
                logits[f"{_target}_logits_group"] = _pred_groups

                # calculate group loss
                _loss_group = self.loss_fn[idx](
                    _pred_groups.view(-1, _groups.shape[-1]),
                    _groups.view(-1, _groups.shape[-1]).float(),
                )

                # add losses together
                _loss = _loss + _loss_group

            # sum losses across all tasks
            loss = loss + _loss

        return (loss, logits) if return_outputs else loss
