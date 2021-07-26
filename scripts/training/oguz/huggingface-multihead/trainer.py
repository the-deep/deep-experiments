import torch.nn as nn
from transformers import Trainer


class MultiHeadTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        groups = inputs.pop("groups")

        if model.iterative:
            pred_groups, pred_labels = model(inputs, gt_groups=groups)

            # calculate group loss
            loss_fn = nn.BCEWithLogitsLoss()
            loss_group = loss_fn(
                pred_groups.view(-1, groups.shape[-1]),
                groups.float().view(-1, groups.shape[-1]),
            )
        else:
            pred_labels = model(inputs)

        # calculate label loss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(
            pred_labels.view(-1, labels.shape[-1]),
            labels.float().view(-1, labels.shape[-1]),
        )
        logits = {"logits": pred_labels}

        # update loss + logits for iterative
        if model.iterative:
            loss = loss + loss_group
            logits["logits_group"] = pred_groups

        return (loss, logits) if return_outputs else loss
