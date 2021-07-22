import torch.nn as nn
from transformers import Trainer


class MultiHeadTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(inputs)

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(
            outputs.view(-1, outputs.shape[-1]),
            labels.float().view(-1, outputs.shape[-1]),
        )

        return (loss, {"logits": outputs}) if return_outputs else loss
