from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
import torchmetrics


class Model(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int):
        super().__init__()
        self.l1 = AutoModel.from_pretrained(model_name_or_path)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, num_labels)

    def forward(self, inputs):
        output = self.l1(
            inputs["ids"],
            attention_mask=inputs["mask"],
        )
        output = output.last_hidden_state
        output = self.l2(output)
        output = self.l3(output)
        return output[:, 0, :]


class SectorsTransformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        pred_threshold: float = 0.5,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = Model(model_name_or_path, num_labels)
        self.pred_threshold = pred_threshold

        self.f1_score_train = torchmetrics.F1(
            num_classes=2,
            threshold=0.5,
            average="macro",
            mdmc_average="samplewise",
            ignore_index=None,
            top_k=None,
            multiclass=True,
            compute_on_step=True,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
        )

        self.f1_score_val = torchmetrics.F1(
            num_classes=2,
            threshold=0.5,
            average="macro",
            mdmc_average="samplewise",
            ignore_index=None,
            top_k=None,
            multiclass=True,
            compute_on_step=True,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
        )

    @auto_move_data
    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = F.binary_cross_entropy_with_logits(outputs, batch["targets"])

        self.f1_score_train(torch.sigmoid(outputs), batch["targets"].to(dtype=torch.long))
        self.log("train_f1", self.f1_score_train, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch)
        val_loss = F.binary_cross_entropy_with_logits(outputs, batch["targets"])

        self.f1_score_val(torch.sigmoid(outputs), batch["targets"].to(dtype=torch.long))
        self.log(
            "val_f1",
            self.f1_score_val,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_nb):
        logits = self(batch)
        preds = torch.sigmoid(logits) > 0.5
        return {"preds": preds, "targets_i": batch["targets"]}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self(batch)
        return {"logits": output}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=1000,  # CHANGE ME
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
