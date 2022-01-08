import os

# setting tokenizers parallelism to false adds robustness when dploying the model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dill import needs to be kept for more robustness in multimodel serialization
import dill

dill.extend(True)


from typing import Optional
from tqdm.auto import tqdm


from torchmetrics.functional import auroc

import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn import metrics

from transformers import AdamW
from transformers.optimization import (
    get_linear_schedule_with_warmup,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import CustomDataset
from utils import flatten, tagname_to_id, get_flat_labels
from architecture import Model
from loss import FocalLoss


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
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-7,
        warmup_steps: int = 500,
        weight_decay: float = 0.1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        dropout_rate: float = 0.3,
        max_len: int = 512,
        output_length: int = 384,
        training_device: str = "cuda",
        keep_neg_examples: bool = False,
        dim_hidden_layer: int = 256,
        **kwargs,
    ):

        super().__init__()
        self.output_length = output_length
        self.column_name = column_name
        self.save_hyperparameters()
        self.targets = train_dataset["target"]
        self.tagname_to_tagid = tagname_to_id(train_dataset["target"])
        self.num_labels = len(self.tagname_to_tagid)
        self.get_first_level_ids()

        self.max_len = max_len
        self.model = Model(
            model_name_or_path,
            self.ids_each_level,
            dropout_rate,
            self.output_length,
            dim_hidden_layer,
        )
        self.tokenizer = tokenizer
        self.val_params = val_params

        self.training_device = training_device

        self.multiclass = multiclass
        self.keep_neg_examples = keep_neg_examples

        self.training_loader = self.get_loaders(
            train_dataset, train_params, self.tagname_to_tagid, self.tokenizer, max_len
        )
        self.val_loader = self.get_loaders(
            val_dataset, val_params, self.tagname_to_tagid, self.tokenizer, max_len
        )
        self.Focal_loss = FocalLoss()

    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        train_loss = self.get_loss(outputs, batch["targets"])

        self.log(
            "train_loss", train_loss.item(), prog_bar=True, on_step=False, on_epoch=True
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        val_loss = self.get_loss(outputs, batch["targets"])
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        return {"val_loss": val_loss}

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

    def get_loaders(
        self, dataset, params, tagname_to_tagid, tokenizer, max_len: int = 128
    ):

        set = CustomDataset(dataset, tagname_to_tagid, tokenizer, max_len)
        loader = DataLoader(set, **params, pin_memory=True)
        return loader

    def get_loss(self, outputs, targets, only_pos: bool = False):

        # keep the if because we want to take negative examples into account for the models that contain
        # no hierarchy (upper level models)
        if len(self.ids_each_level) == 1:
            return self.Focal_loss(outputs[0], targets)
        else:
            tot_loss = 0
            for i_th_level in range(len(self.ids_each_level)):
                ids_one_level = self.ids_each_level[i_th_level]
                outputs_i_th_level = outputs[i_th_level]
                targets_one_level = targets[:, ids_one_level]
                # main objective: for each level, if row contains only zeros, not to do backpropagation

                if only_pos:
                    mask_ids_neg_example = [
                        not bool(int(torch.sum(one_row)))
                        for one_row in targets_one_level
                    ]
                    outputs_i_th_level[mask_ids_neg_example, :] = 1e-8

                tot_loss += self.Focal_loss(outputs_i_th_level, targets_one_level)

            return tot_loss

    def get_first_level_ids(self):
        all_names = list(self.tagname_to_tagid.keys())
        if np.all(["->" in name for name in all_names]):
            first_level_names = list(
                np.unique([name.split("->")[0] for name in all_names])
            )
            self.ids_each_level = [
                [i for i in range(len(all_names)) if name in all_names[i]]
                for name in first_level_names
            ]

        else:
            self.ids_each_level = [[i for i in range(len(all_names))]]

    def custom_predict(
        self, validation_dataset, testing=False, hypertuning_threshold: bool = False
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
                logits = torch.cat(logits, dim=1)  # have a matrix like in the beginning
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

        data_for_threshold_tuning = self.val_loader.dataset.data
        indexes, logit_predictions, y_true, _ = self.custom_predict(
            data_for_threshold_tuning, hypertuning_threshold=True
        )

        thresholds_list = np.linspace(0.0, 1.0, 101)[::-1]
        optimal_thresholds_dict = {}
        optimal_scores = []
        for ids_one_level in self.ids_each_level:
            y_true_one_level = y_true[:, ids_one_level]
            logit_preds_one_level = logit_predictions[:, ids_one_level]

            """if len(self.ids_each_level) > 1: #multitask

                mask_at_least_one_pos = [bool(sum(row)) for row in y_true_one_level]
                threshold_tuning_gt = y_true_one_level[mask_at_least_one_pos]
                threshold_tuning_logit_preds = logit_preds_one_level[mask_at_least_one_pos]
            else: #no multitask
                threshold_tuning_gt = y_true_one_level
                threshold_tuning_logit_preds = logit_predictions

            assert(threshold_tuning_logit_preds.shape == threshold_tuning_gt.shape)"""

            for j in range(len(ids_one_level)):
                scores = []
                for thresh_tmp in thresholds_list:
                    metric = self.get_metric(
                        logit_preds_one_level,
                        y_true_one_level,
                        beta_f1,
                        j,
                        thresh_tmp,
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
                    list(self.tagname_to_tagid.keys())[ids_one_level[j]]
                ] = max_threshold

        self.optimal_thresholds = optimal_thresholds_dict

        return np.mean(optimal_scores)

    def get_metric(self, preds, groundtruth, beta_f1, column_idx, threshold_tmp):
        columns_logits = np.array(preds[:, column_idx])

        column_pred = [
            1 if one_logit > threshold_tmp else 0 for one_logit in columns_logits
        ]

        if self.multiclass:
            metric = metrics.fbeta_score(
                groundtruth[:, column_idx],
                column_pred,
                beta_f1,
                average="macro",
            )
        else:
            metric = metrics.f1_score(
                groundtruth[:, column_idx],
                column_pred,
                average="macro",
            )
        return metric
