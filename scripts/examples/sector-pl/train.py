# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import sys

sys.path.append(".")

import os
import logging
import argparse
from pathlib import Path
from typing import Optional

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from transformers import AutoTokenizer
from transformers import AdamW, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
import torchmetrics
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


class SectorsDataset(Dataset):
    def __init__(self, dataframe, sectorname_to_sectorid, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.excerpt_text = dataframe["excerpt"].tolist() if dataframe is not None else None
        self.targets = dataframe["sectors"].tolist() if dataframe is not None else None
        self.sectorname_to_sectorid = sectorname_to_sectorid
        self.sectorid_to_sectorname = list(sectorname_to_sectorid.keys())
        self.max_len = max_len

    def encode_example(self, excerpt_text: str, index=None, as_batch: bool = False):
        # excerpt_text = " ".join(excerpt_text.split())

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
        targets = None
        if self.targets:
            target_indices = [
                self.sectorname_to_sectorid[target]
                for target in self.targets[index]
                if target in self.sectorname_to_sectorid
            ]
            targets = np.zeros(len(self.sectorname_to_sectorid), dtype=np.int)
            targets[target_indices] = 1

        encoded = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.float32) if targets is not None else None,
        }
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


class SectorsDatasetPreds(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=200):
        self.tokenizer = tokenizer
        self.excerpt_text = dataframe["excerpt"].tolist() if dataframe is not None else None
        self.max_len = max_len

    def encode_example(self, excerpt_text: str):
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
        return encoded

    def __len__(self):
        return len(self.excerpt_text)

    def __getitem__(self, index):
        excerpt_text = str(self.excerpt_text[index])
        return self.encode_example(excerpt_text)


class TransformersQAWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model.eval()
        super().__init__()

    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        dataset = SectorsDatasetPreds(model_input, self.tokenizer)
        val_params = {"batch_size": 16, "shuffle": False, "num_workers": 0}
        dataloader = DataLoader(dataset, **val_params)
        with torch.no_grad():
            predictions = [model.forward(batch) for batch in dataloader]
        predictions = torch.cat(predictions)
        predictions = predictions.argmax(1)
        return pd.Series(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # MLflow related parameters
    parser.add_argument("--tracking_uri", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--classes", type=str)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    args, _ = parser.parse_known_args()

    logging.info("reading data")
    train_df = pd.read_pickle(f"{args.train}/train.pickle")
    val_df = pd.read_pickle(f"{args.train}/val.pickle")
    classes = eval(args.classes)
    class_to_id = {class_: i for i, class_ in enumerate(classes)}

    logging.info("building training and testing datasets")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    training_set = SectorsDataset(train_df, class_to_id, tokenizer, args.max_len)
    val_set = SectorsDataset(val_df, class_to_id, tokenizer, args.max_len)

    # set remote mlflow server
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run():

        train_params = {"batch_size": args.train_batch_size, "shuffle": True, "num_workers": 0}
        val_params = {"batch_size": args.eval_batch_size, "shuffle": False, "num_workers": 0}

        training_loader = DataLoader(training_set, **train_params)
        val_loader = DataLoader(val_set, **val_params)

        params = {
            "train": train_params,
            "val": val_params,
        }
        mlflow.log_params(params)

        logging.info("training model")

        trainer = pl.Trainer(
            # callbacks=[early_stopping_callback, checkpoint_callback],
            gpus=1,
            max_epochs=args.epochs,
        )
        model = SectorsTransformer(
            args.model_name,
            len(class_to_id),
        )
        trainer.fit(model, training_loader, val_loader)

        # Log
        requirement_file = str(Path(__file__).parent / "requirements.txt")
        with open(requirement_file, "r") as f:
            requirements = f.readlines()
        requirements = [x.replace("\n", "") for x in requirements]

        default_env = mlflow.pytorch.get_default_conda_env()
        pip_dependencies = default_env["dependencies"][2]["pip"]
        pip_dependencies.extend(requirements)

        prediction_wrapper = TransformersQAWrapper(tokenizer, model)
        mlflow.pyfunc.log_model(
            python_model=prediction_wrapper, artifact_path="model", conda_env=default_env
        )

        # ABS ERROR AND LOG COUPLE PERF METRICS
        # logging.info("evaluating model")
        # df_metrics_val = model.custom_eval(val_loader, class_to_id)
        # logging.info("metric_val", df_metrics_val.shape)

        # mlflow.pytorch.log_model(
        #     model.model.cpu(),
        #     artifact_path=args.model_name,
        #     registered_model_name="pytorch-first-example",
        # )
