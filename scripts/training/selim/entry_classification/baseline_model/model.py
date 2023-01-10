import os

# setting tokenizers parallelism to false adds robustness when dploying the model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dill import needs to be kept for more robustness in multimodel serialization
import dill

dill.extend(True)


from typing import Optional
from tqdm.auto import tqdm

import pytorch_lightning as pl

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn import metrics

from transformers import (
    AdamW,
    AutoModel,
)
from transformers.optimization import (
    get_linear_schedule_with_warmup,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import CustomDataset

from utils import compute_weights, tagname_to_id
from architecture import Model


class Transformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        train_dataset,
        val_dataset,
        train_params,
        val_params,
        tokenizer,
        column_name,
        multiclass,
        dim_hidden_layer: int,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 500,
        weight_decay: float = 0.1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        dropout_rate: float = 0.3,
        max_len: int = 512,
        output_length: int = 384,
        training_device: str = "cuda",
        **kwargs,
    ):

        super().__init__()
        self.output_length = output_length
        self.column_name = column_name
        self.save_hyperparameters()
        self.targets = train_dataset["target"]
        self.tagname_to_tagid = tagname_to_id(train_dataset["target"])
        self.num_labels = len(self.tagname_to_tagid)
        self.max_len = max_len
        self.model = Model(
            model_name_or_path,
            self.num_labels,
            dropout_rate,
            self.output_length,
            dim_hidden_layer,
        )
        self.tokenizer = tokenizer
        self.val_params = val_params

        self.training_device = training_device

        self.weight_classes = self.get_weights()
        self.weight_classes = torch.tensor(self.weight_classes).to(self.training_device)

        self.multiclass = multiclass

        self.training_loader = self.get_loaders(
            train_dataset, train_params, self.tagname_to_tagid, self.tokenizer, max_len
        )
        self.val_loader = self.get_loaders(
            val_dataset, val_params, self.tagname_to_tagid, self.tokenizer, max_len
        )

    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        train_loss = F.binary_cross_entropy_with_logits(
            outputs, batch["targets"], weight=self.weight_classes
        )

        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        val_loss = F.binary_cross_entropy_with_logits(outputs, batch["targets"])

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        return {"val_loss": val_loss}

    def predict_step(self, batch, batch_idx):
        output = self(batch)
        return {"logits": output}

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        self.dataset_size = len(self.train_dataloader().dataset)
        num_devices = max(1, self.hparams.gpus)
        effective_batch_size = (
            self.hparams.train_batch_size
            * self.hparams.accumulate_grad_batches
            * num_devices
        )
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        # model = self.model
        """no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = ReduceLROnPlateau(
            optimizer, "min", 0.5, patience=self.hparams.max_epochs // 6
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.training_loader

    def val_dataloader(self):
        return self.val_loader

    def get_weights(self) -> list:

        list_tags = list(self.tagname_to_tagid.keys())
        number_data_classes = []
        for tag in list_tags:
            nb_data_in_class = self.targets.apply(lambda x: tag in (x)).sum()
            number_data_classes.append(nb_data_in_class)
        weights = compute_weights(number_data_classes, self.targets.shape[0])
        weights = [weight if weight < 1 else weight ** 2 for weight in weights]
        return weights

    def get_loaders(
        self,
        dataset,
        params,
        tagname_to_tagid,
        tokenizer,
        max_len: int = 128,
        drop_last: bool = True,
    ):

        set = CustomDataset(dataset, tagname_to_tagid, tokenizer, max_len)
        loader = DataLoader(set, **params, pin_memory=True, drop_last=drop_last)
        return loader

    def custom_predict(
        self, validation_dataset, testing=False, hypertuning_threshold=False
    ):
        """
        1) get raw predictions
        2) postprocess them to output an output compatible with what we want in the inference
        """

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if testing:
            self.val_params["num_workers"] = 0

        validation_loader = self.get_loaders(
            validation_dataset,
            self.val_params,
            self.tagname_to_tagid,
            self.tokenizer,
            self.max_len,
            drop_last=False,
        )

        if torch.cuda.is_available():
            testing_device = "cuda"
        else:
            testing_device = "cpu"

        self.to(testing_device)
        self.eval()
        self.freeze()
        y_true = []
        logit_predictions = []
        indexes = []

        with torch.no_grad():
            for batch in tqdm(
                validation_loader,
                total=len(validation_loader.dataset) // validation_loader.batch_size,
            ):

                if not testing:
                    y_true.append(batch["targets"].detach())
                    indexes.append(batch["entry_id"].detach())

                logits = self(
                    {
                        "ids": batch["ids"].to(testing_device),
                        "mask": batch["mask"].to(testing_device),
                        "token_type_ids": batch["token_type_ids"].to(testing_device),
                    }
                )

                logits_to_array = np.array([np.array(t) for t in logits.cpu()])
                logit_predictions.append(logits_to_array)

        logit_predictions = np.concatenate(logit_predictions)
        logit_predictions = sigmoid(logit_predictions)

        target_list = list(self.tagname_to_tagid.keys())
        probabilities_dict = []
        # postprocess predictions
        for i in range(logit_predictions.shape[0]):

            # Return predictions
            # row_pred = np.array([0] * self.num_labels)
            row_logits = logit_predictions[i, :]

            # Return probabilities
            probabilities_item_dict = {}
            for j in range(logit_predictions.shape[1]):
                if hypertuning_threshold:
                    probabilities_item_dict[target_list[j]] = row_logits[j]
                else:
                    probabilities_item_dict[target_list[j]] = (
                        row_logits[j] / self.optimal_thresholds[target_list[j]]
                    )

            probabilities_dict.append(probabilities_item_dict)

        if not testing:
            y_true = np.concatenate(y_true)
            indexes = np.concatenate(indexes)
            return indexes, logit_predictions, y_true, probabilities_dict

        else:
            return probabilities_dict

    def hypertune_threshold(self, beta_f1: float = 0.8):
        """
        having the probabilities, loop over a list of thresholds to see which one:
        1) yields the best results
        2) without being an aberrant value
        """

        thresholds_list = np.linspace(0.0, 1.0, 101)[::-1]
        data_for_threshold_tuning = self.val_loader.dataset.data
        indexes, logit_predictions, y_true, _ = self.custom_predict(
            data_for_threshold_tuning, hypertuning_threshold=True
        )
        optimal_thresholds_dict = {}
        optimal_scores = []

        for j in range(logit_predictions.shape[1]):
            scores = []
            for thresh_tmp in thresholds_list:

                columns_logits = np.array(logit_predictions[:, j])

                column_pred = [
                    1 if columns_logits[i] > thresh_tmp else 0
                    for i in range(logit_predictions.shape[0])
                ]

                if self.multiclass:
                    metric = metrics.fbeta_score(
                        y_true[:, j], column_pred, beta_f1, average="macro"
                    )
                else:
                    metric = metrics.f1_score(
                        y_true[:, j], column_pred, average="macro"
                    )

                scores.append(metric)

            max_threshold = 0
            max_score = 0
            for i in range(1, len(scores) - 1):
                score = np.mean(scores[i - 1 : i + 2])
                if score >= max_score:
                    max_score = score
                    max_threshold = thresholds_list[i]

            optimal_scores.append(max_score)
            optimal_thresholds_dict[
                list(self.tagname_to_tagid.keys())[j]
            ] = max_threshold

        self.optimal_thresholds = optimal_thresholds_dict

        return np.mean(optimal_scores)