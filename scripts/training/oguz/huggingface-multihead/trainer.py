from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    DataCollator,
    Dataset,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from utils import sigmoid_focal_loss


class MultiHeadTrainer(Trainer):
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
        focal_loss: bool = False,
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
        self.loss_fn = sigmoid_focal_loss if focal_loss else torch.nn.BCEWithLogitsLoss()

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
