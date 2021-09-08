from typing import Optional

from tqdm.auto import tqdm

import torchmetrics
from torchmetrics.functional import auroc

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
from utils import (
    compute_weights,
    tagname_to_id
)

class CustomDataset(Dataset):
    def __init__(self, dataframe, tagname_to_tagid, tokenizer, max_len: int = 150):
        self.tokenizer = tokenizer
        self.data = dataframe

        if dataframe is None:
            self.excerpt_text = None
        elif type(dataframe) is pd.Series:
            self.excerpt_text = dataframe.tolist()
        else:
            self.excerpt_text = dataframe["excerpt"].tolist()

        try:
            self.targets = list(dataframe["target"])
            self.entry_ids = list(dataframe["entry_id"])
        except Exception:
            self.targets = None
            self.entry_ids = None

        self.tagname_to_tagid = tagname_to_tagid
        self.tagid_to_tagname = list(tagname_to_tagid.keys())
        self.max_len = max_len

    def encode_example(self, excerpt_text: str, index=None, as_batch: bool = False):

        inputs = self.tokenizer(
            excerpt_text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        encoded = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

        targets = None
        if self.targets:
            target_indices = [
                self.tagname_to_tagid[target]
                for target in self.targets[index]
                if target in self.tagname_to_tagid
            ]
            targets = np.zeros(len(self.tagname_to_tagid), dtype=np.int)
            targets[target_indices] = 1

            encoded["targets"] = (
                torch.tensor(targets, dtype=torch.float32) if targets is not None else None
            )
            encoded["entry_id"] = self.entry_ids[index]

        if as_batch:
            return {
                "ids": encoded["ids"].unsqueeze(0),
                "mask": encoded["mask"].unsqueeze(0),
                "token_type_ids": encoded["ids"].unsqueeze(0),
            }
        return encoded

    def __len__(self):
        return len(self.excerpt_text)

    def __getitem__(self, index):
        excerpt_text = str(self.excerpt_text[index])
        return self.encode_example(excerpt_text, index)


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
        weight_classes = self.get_weights()
        self.weight_classes = torch.tensor(weight_classes).to("cuda")
        # else:
        #     self.use_weights = False

        self.multiclass = multiclass
        self.empty_dataset = CustomDataset(None, self.tagname_to_tagid, tokenizer, max_len)
        self.pred_threshold = pred_threshold
        self.training_loader = self.get_loaders(
            train_dataset, train_params, self.tagname_to_tagid, self.tokenizer, max_len
        )
        self.val_loader = self.get_loaders(
            val_dataset, val_params, self.tagname_to_tagid, self.tokenizer, max_len
        )

        if multiclass:
            self.f1_score_train = torchmetrics.F1(
                num_classes=2,
                threshold=self.pred_threshold,
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
                threshold=self.pred_threshold,
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

        else:
            self.f1_score_train = torchmetrics.F1()

            self.f1_score_val = torchmetrics.F1()

    def get_weights(self)->list:

        list_tags = list(self.tagname_to_tagid.keys())

        number_data_classes = []
        for tag in list_tags:
            nb_data_in_class = self.targets.apply(lambda x: tag in (x)).sum()
            number_data_classes.append(nb_data_in_class)

        weights = compute_weights(number_data_classes, self.targets.shape[0])

        weights = [weight if weight < 5 else weight ** 2 for weight in weights]
        return weights

    @auto_move_data
    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        if self.use_weights:
            loss = F.binary_cross_entropy_with_logits(
                outputs, batch["targets"], weight=self.weight_classes
            )
        else:
            loss = F.binary_cross_entropy_with_logits(outputs, batch["targets"])

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
        preds = torch.sigmoid(logits) >= self.pred_threshold
        pred_classes = []
        for pred in preds:
            pred_classes_i = [
                self.empty_dataset.sectorid_to_sectorname[i] for i, p in enumerate(pred) if p
            ]
            pred_classes.append(pred_classes_i)
        self.log({"pred_classes": pred_classes})

    def custom_predict(self, validation_dataset, return_all=False, testing=False):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if testing:
            self.val_params["num_workers"] = 0

        validation_loader = self.get_loaders(
            validation_dataset, self.val_params, self.tagname_to_tagid, self.tokenizer, self.max_len
        )
        if self.device.type == "cpu":
            self.to("cuda")
        self.eval()
        self.freeze()
        indexes = []
        y_true = []
        logit_predictions = []

        if testing:
            with torch.no_grad():
                for batch in tqdm(
                    validation_loader,
                    total=len(validation_loader.dataset) // validation_loader.batch_size,
                ):

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
            target_list = list(self.tagname_to_tagid.keys())
            probabilities_dict = []
            # postprocess predictions
            for i in range(logit_predictions.shape[0]):

                # Return predictions
                # row_pred = np.array([0] * self.num_labels)
                row_logits = logit_predictions[i, :]

                # predictions.append(list(list_tags[row_logits > self.pred_threshold]))

                # Return probabilities
                probabilities_item_dict = {}
                for j in range(logit_predictions.shape[1]):
                    probabilities_item_dict[target_list[j]] = row_logits[j]

                probabilities_dict.append(probabilities_item_dict)

            # df_returned = pd.DataFrame(predictions, columns=['predictions_2d_subpillars'])
            # df_returned['probabilities_2d_subpillars'] = probabilities_dict

            return probabilities_dict

        else:

            with torch.no_grad():
                for batch in tqdm(
                    validation_loader,
                    total=len(validation_loader.dataset) // validation_loader.batch_size,
                ):

                    logits = self(
                        {
                            "ids": batch["ids"].to("cuda"),
                            "mask": batch["mask"].to("cuda"),
                            "token_type_ids": batch["token_type_ids"].to("cuda"),
                        }
                    )

                    y_true.append(batch["targets"].numpy().astype(np.int))
                    indexes.append(batch["entry_id"].numpy().astype(np.int))

                    logits_to_array = np.array([np.array(t) for t in logits.cpu()])
                    logit_predictions.append(logits_to_array)

            y_true = np.concatenate(y_true)
            indexes = np.concatenate(indexes)
            logit_predictions = np.concatenate(logit_predictions)

            logit_predictions = sigmoid(logit_predictions)
            predictions = []
            y_true_final = []
            indexes_final = []
            # postprocess predictions
            for i in range(logit_predictions.shape[0]):
                row_pred = np.array([0] * self.num_labels)
                row_logits = logit_predictions[i, :]

                if np.any(row_logits >= self.pred_threshold):
                    if self.multiclass:
                        row_pred[row_logits > self.pred_threshold] = 1
                    else:
                        row_pred[np.argmax(row_logits)] = 1

                    predictions.append(row_pred)
                    y_true_final.append(y_true[i])
                    indexes_final.append(indexes[i])

            if return_all:
                return (
                    np.array(predictions),
                    np.array(y_true_final),
                    zip(indexes, logit_predictions, y_true),
                )

            return np.array(predictions), np.array(y_true_final)

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
            num_training_steps=self.total_steps(),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


    def train_dataloader(self):
        return self.training_loader

    def val_dataloader(self):
        return self.val_loader

    def compute_metrics(self, preds_val_all, y_true, tagname_to_tagid):
        f1_scores = []
        recalls = []
        precisions = []
        accuracies = []

        for tag_name, tag_id in tagname_to_tagid.items():
            cls_rprt = metrics.classification_report(
                y_true[:, tag_id], preds_val_all[:, tag_id], output_dict=True
            )
            precisions.append(cls_rprt["macro avg"]["precision"])
            recalls.append(cls_rprt["macro avg"]["recall"])
            f1_scores.append(cls_rprt["macro avg"]["f1-score"])
            accuracies.append(cls_rprt["accuracy"])

        metrics_df = pd.DataFrame(
            {
                "Sector": list(tagname_to_tagid.keys()),
                "Precision": precisions,
                "Recall": recalls,
                "F1 Score": f1_scores,
                "Accuracy": accuracies,
            }
        )
        metrics_df.loc["mean"] = metrics_df.mean()
        return metrics_df

    def get_results_pillar_from_subpillar(self, preds_val_all, y_true):
        list_tags_subpillars = list(self.tagname_to_tagid.keys())
        list_tags_pillars = sorted(
            list(set([tags.split("->")[0] for tags in list_tags_subpillars]))
        )
        pillars_tagname_to_tagid = {tag: i for i, tag in enumerate(list(sorted(list_tags_pillars)))}

        n_subpillars_per_pillar = [
            len(list(filter(lambda x: pillar in x, list_tags_subpillars)))
            for pillar in list_tags_pillars
        ]

        def subpillars_to_pillars(y):
            print(y.shape)
            pillars_true_y = []
            for row_nb in range(y.shape[0]):
                row = y[row_nb, :]
                nb_pillars = len(n_subpillars_per_pillar)
                result = np.array([0] * nb_pillars)
                count = 0
                for pillar_nb in range(nb_pillars):

                    supillar_nb_tmp = n_subpillars_per_pillar[pillar_nb]

                    if np.any(row[count : count + supillar_nb_tmp]):
                        result[pillar_nb] = 1

                    count += supillar_nb_tmp

                if np.any(row[count:]):
                    result[-1] = 1

                pillars_true_y.append(result)

            return np.array(pillars_true_y)

        y_true_pillars = subpillars_to_pillars(y_true)
        preds_val_pillars = subpillars_to_pillars(preds_val_all)

        return self.compute_metrics(preds_val_pillars, y_true_pillars, pillars_tagname_to_tagid)

    def custom_eval(self, validation_dataset):

        preds_val_all, y_true = self.custom_predict(validation_dataset)
        ratio_evaluated_sentences = len(y_true) * 100 / validation_dataset.shape[0]
        metrics_subpillars = self.compute_metrics(preds_val_all, y_true, self.tagname_to_tagid)
        #metrics_pillars = self.get_results_pillar_from_subpillar(preds_val_all, y_true)

        return metrics_subpillars, ratio_evaluated_sentences
