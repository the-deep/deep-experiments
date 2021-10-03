from typing import Optional
import timeit
from tqdm.auto import tqdm
import re
import timeit
import dill

import torchmetrics
from torchmetrics.functional import auroc

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data

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

from data import CustomDataset

from utils import (
    compute_weights,
    tagname_to_id
)

class Model(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        token_len: int,
        dropout_rate=0.3,
        output_length=384,
    ):
        super().__init__()
        self.l0 = AutoModel.from_pretrained(model_name_or_path)
        # batch normlisation to be integrated
        self.l1 = torch.nn.Linear(output_length, token_len)
        self.l2 = torch.nn.BatchNorm1d(token_len)
        self.l3 = torch.nn.Dropout(dropout_rate)
        self.l4 = torch.nn.Linear(token_len, num_labels)

    def forward(self, inputs):
        output = self.l0(
            inputs["ids"],
            attention_mask=inputs["mask"],
        )
        output = output.last_hidden_state
        output = F.selu(self.l1(output))
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        return output[:, 0, :]


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
        multiclass=True,
        pred_threshold: float = 0.5,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 500,
        weight_decay: float = 0.1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        dropout_rate: float = 0.3,
        max_len: int = 150,
        output_length=384,
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
            model_name_or_path, self.num_labels, max_len, dropout_rate, self.output_length
        )
        self.tokenizer = tokenizer
        self.val_params = val_params

        # if any(weight_classes):
        self.use_weights = True
        self.weight_classes = self.get_weights()
        self.weight_classes = torch.tensor(self.weight_classes).to("cuda")

        self.multiclass = multiclass
        self.empty_dataset = CustomDataset(None, self.tagname_to_tagid, tokenizer, max_len)
        self.threshold = pred_threshold
        self.training_loader = self.get_loaders(
            train_dataset, train_params, self.tagname_to_tagid, self.tokenizer, max_len
        )
        self.val_loader = self.get_loaders(
            val_dataset, val_params, self.tagname_to_tagid, self.tokenizer, max_len
        )

        #if multiclass:
        self.f1_score_train = torchmetrics.F1(
            num_classes=2,
            threshold=self.threshold,
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
            threshold=self.threshold,
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

        """else:
            self.f1_score_train = torchmetrics.F1()
            self.f1_score_val = torchmetrics.F1()"""

    def get_weights(self)->list:

        list_tags = list(self.tagname_to_tagid.keys())

        number_data_classes = []
        for tag in list_tags:
            nb_data_in_class = self.targets.apply(lambda x: tag in (x)).sum()
            number_data_classes.append(nb_data_in_class)

        weights = compute_weights(number_data_classes, self.targets.shape[0])

        weights = [weight if weight < 1 else weight ** 2 for weight in weights]
        return weights

    @auto_move_data
    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = F.binary_cross_entropy_with_logits(
            outputs, batch["targets"], weight=self.weight_classes
        )
        
        if self.multiclass:
            self.f1_score_train(torch.sigmoid(outputs), batch["targets"].to(dtype=torch.long))
        else:
            argmax = outputs.argmax(1)
            processed_output = torch.zeros(outputs.shape).scatter(1, argmax.unsqueeze(1), 1.0)
            self.f1_score_train(processed_output, batch["targets"].to(dtype=torch.long))

        self.log("train_f1", self.f1_score_train, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch)
        val_loss = F.binary_cross_entropy_with_logits(outputs, batch["targets"])

        if self.multiclass:
            self.f1_score_val(torch.sigmoid(outputs), batch["targets"].to(dtype=torch.long))
        else:
            argmax = outputs.argmax(1)
            processed_output = torch.zeros(outputs.shape).scatter(1, argmax.unsqueeze(1), 1.0)
            self.f1_score_val(processed_output, batch["targets"].to(dtype=torch.long))

        self.log(
            "val_f1", self.f1_score_val, on_step=True, on_epoch=True, prog_bar=True, logger=False
        )

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return {"val_loss": val_loss, "val_f1": self.f1_score_val}

    def test_step(self, batch, batch_nb):
        logits = self(batch)
        preds = torch.sigmoid(logits) > 0.5
        return {"preds": preds, "targets_i": batch["targets"]}

    def on_test_epoch_end(self, outputs):
        preds = torch.cat([output["preds"] for output in outputs]).cpu()
        targets = torch.cat([output["targets_i"] for output in outputs]).cpu()

        for i in range(targets.shape[1]):
            class_roc_auc = auroc(preds[:, i], targets[:, i])
            self.log(f"{self.empty_dataset.sectorid_to_sectorname[i]}_roc_auc/Train", class_roc_auc)
            class_f1 = metrics.f1_score(targets[:, i], preds[:, i])
            self.log(f"{self.empty_dataset.sectorid_to_sectorname[i]}_f1/Train", class_f1)

    def get_loaders(self, dataset, params, tagname_to_tagid, tokenizer, max_len: int = 128):

        set = CustomDataset(dataset, tagname_to_tagid, tokenizer, max_len)

        loader = DataLoader(set, **params)
        return loader

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self(batch)
        return {"logits": output}

    def on_predict_epoch_end(self, outputs):
        logits = torch.cat([output["logits"] for output in outputs[0]])
        preds = torch.sigmoid(logits) >= self.threshold
        pred_classes = []
        for pred in preds:
            pred_classes_i = [
                self.empty_dataset.sectorid_to_sectorname[i] for i, p in enumerate(pred) if p
            ]
            pred_classes.append(pred_classes_i)
        self.log({"pred_classes": pred_classes})
    
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        self.dataset_size = len(self.train_dataloader().dataset)
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = (
            self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        )
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        #model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
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
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps(),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


    def train_dataloader(self):
        return self.training_loader

    def val_dataloader(self):
        return self.val_loader

    def custom_predict(self, validation_dataset, testing=False):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))  

        if testing:
            self.val_params["num_workers"] = 0

        validation_loader = self.get_loaders(
            validation_dataset, self.val_params, self.tagname_to_tagid, self.tokenizer, self.max_len
        )

        self.to("cuda")
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
                    y_true.append(batch["targets"].numpy().astype(np.int))
                    indexes.append(batch["entry_id"].numpy().astype(np.int))

                logits = self(
                    {
                        "ids": batch["ids"].to("cuda"),
                        "mask": batch["mask"].to("cuda"),
                        "token_type_ids": batch["token_type_ids"].to("cuda"),
                    }
                )

                logits_to_array = np.array([np.array(t) for t in logits.cpu()])
                logit_predictions.append(logits_to_array)

        logit_predictions = np.concatenate(logit_predictions)
        logit_predictions = sigmoid(logit_predictions)

        if not testing:
            y_true = np.concatenate(y_true)
            indexes = np.concatenate(indexes)
            return indexes, logit_predictions, y_true

        else:
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
                    probabilities_item_dict[target_list[j]] = row_logits[j]

                probabilities_dict.append(probabilities_item_dict)

            return probabilities_dict
            

    def hypertune_threshold (
        self, 
        beta_f1:float=0.5):

        thresholds_list = np.linspace(0.0, 1.0, 101)

        start = timeit.default_timer()

        data_for_threshold_tuning = self.val_loader.dataset.data

        indexes, logit_predictions, y_true  = self.custom_predict(data_for_threshold_tuning)
        stop = timeit.default_timer()
        time_for_predictions = stop - start

        optimal_thresholds_dict = {}

        for j in range (logit_predictions.shape[1]):
            max_threshold = 0
            max_f1 = 0

            for thresh_tmp in thresholds_list:
                
                columns_logits = np.array(logit_predictions[:,j])

                column_pred = [
                    1 if columns_logits[i]> thresh_tmp else 0 for i in range (logit_predictions.shape[0])
                    ]

                custom_f1 = metrics.fbeta_score(y_true[:,j], column_pred, beta_f1, average='macro')

                if custom_f1>max_f1:
                    max_f1 = custom_f1
                    max_threshold = thresh_tmp
                    
            optimal_thresholds_dict[list(self.tagname_to_tagid.keys())[j]] = max_threshold

        self.optimal_thresholds = optimal_thresholds_dict

        optimal_metrics = self.custom_eval(logit_predictions, y_true, beta_f1)

        results = {
            'indexes': indexes,
            'logit_predictions': logit_predictions,
            'groundtruth': y_true,
            'thresholds': optimal_thresholds_dict,
            'optimal_metrics': optimal_metrics
        }

        return time_for_predictions, results


    def custom_eval(self, logit_predictions, y_true, beta_f1):

        results_dict = {}
        overall_recall = []
        overall_precision = []
        overall_custom_f1 = []

        # postprocess predictions
        for j in range(logit_predictions.shape[1]):
            thresh_value_tmp = list(self.optimal_thresholds.values())[j]
            sub_tag_name = list(self.optimal_thresholds.keys())[j]
            probas_column = logit_predictions[:, j]
            true_preds_column = y_true[:, j]
            
            if self.multiclass:
                column_pred = [
                    1 if pred>thresh_value_tmp else 0 for pred in probas_column
                    ]
            else:
                column_pred = [
                    1 if i==np.argmax(probas_column) else 0 for i in range (len(probas_column))
                    ]

            precision = metrics.precision_score(true_preds_column, column_pred, average='macro')
            custom_f1 = metrics.fbeta_score(true_preds_column, column_pred, beta_f1, average='macro')
            recall = metrics.recall_score(true_preds_column, column_pred, average='macro')

            overall_recall.append(recall)
            overall_precision.append(precision)
            overall_custom_f1.append(custom_f1)

            cleaned_name = re.sub("[^0-9a-zA-Z]+", "_", sub_tag_name)

            results_dict['precision_'+cleaned_name] = precision
            results_dict['recall_'+cleaned_name] = recall
            results_dict['beta_f1_'+cleaned_name] = custom_f1

        results_dict['overall_custom_f1_'+self.column_name] = np.mean(overall_custom_f1)
        results_dict['overall_precision_'+self.column_name] = np.mean(overall_precision)
        results_dict['overall_recall_'+self.column_name] = np.mean(overall_recall)

        return results_dict