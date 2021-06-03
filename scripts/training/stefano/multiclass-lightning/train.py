import sys
import os
import argparse
import logging
import pickle
from pathlib import Path
import random

import pandas as pd
import torch
import pytorch_lightning as pl
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--n_classes", type=int, default=10)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_df = pd.read_pickle(f"{args.training_dir}/train_df.pickle")
    val_df = pd.read_pickle(f"{args.training_dir}/val_df.pickle")
    test_df = pd.read_pickle(f"{args.training_dir}/test_df.pickle")
    logger.info(f" loaded train_dataset shape is: {train_df.shape}")
    logger.info(f" loaded val_dataset shape is: {val_df.shape}")
    logger.info(f" loaded test_dataset shape is: {test_df.shape}")

    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))

    def compute_metrics(outputs, labels):
        labels = labels.detach().cpu().long()
        outputs = sigmoid(outputs.detach().cpu())
        outputs = (outputs > 0.5).long()

        logging.info(
            {
                "f1": float(f1_score(labels, outputs)),
                "stupid_metric": random.random(),
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    class TransformerDataset(Dataset):
        def __init__(self, tokenizer, df):
            self.tokenizer = tokenizer
            self.labels = list(df["labels"])
            self.texts = list(df["texts"])

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            inputs = self.tokenizer.encode_plus(
                text,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {
                "input_ids": inputs["input_ids"].flatten(),
                "attention_mask": inputs["attention_mask"].flatten(),
                "label": torch.tensor(self.labels[idx], dtype=torch.float),
            }
            return inputs

    class DataModule(pl.LightningDataModule):
        def __init__(
            self,
            train_df,
            val_df,
            test_df,
            tokenizer,
            train_batch_size=16,
            eval_batch_size=64,
            max_token_len=200,
        ):
            super().__init__()
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df
            self.tokenizer = tokenizer
            self.train_batch_size = train_batch_size
            self.eval_batch_size = eval_batch_size
            self.max_token_len = max_token_len

        def setup(self, stage=None):
            self.train_dataset = TransformerDataset(tokenizer=self.tokenizer, df=self.train_df)
            self.val_dataset = TransformerDataset(tokenizer=self.tokenizer, df=self.val_df)
            self.test_dataset = TransformerDataset(tokenizer=self.tokenizer, df=self.test_df)

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size=self.eval_batch_size)

        def test_dataloader(self):
            return DataLoader(self.test_dataset, batch_size=self.eval_batch_size)

    # we will use the BERT base model(the smaller one)
    class DistilClassifier(pl.LightningModule):
        # Set up the classifier
        def __init__(
            self,
            n_classes=10,
            steps_per_epoch=None,
            n_epochs=1,
            lr=2e-5,
            compute_metrics=compute_metrics,
        ):
            super().__init__()

            self.model = AutoModel.from_pretrained(args.model_name)
            self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
            self.steps_per_epoch = steps_per_epoch
            self.n_epochs = n_epochs
            self.lr = lr
            self.criterion = nn.BCEWithLogitsLoss()
            self.compute_metrics = compute_metrics

        def forward(self, batch):
            output = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            pooler_output = output.last_hidden_state[:, 0, :]
            output = self.classifier(pooler_output)

            return output

        def training_step(self, batch, batch_idx):
            labels = batch["label"]

            outputs = self(batch)
            loss = self.criterion(outputs, labels)
            self.log("train_loss", loss, on_step=True, on_epoch=False)
            self.compute_metrics(outputs, labels)

            return {"loss": loss, "predictions": outputs, "labels": labels}

        def validation_step(self, batch, batch_idx):
            labels = batch["label"]

            outputs = self(batch)
            loss = self.criterion(outputs, labels)
            self.log("val_loss", loss, on_step=True)

            return loss

        def test_step(self, batch, batch_idx):
            labels = batch["label"]

            outputs = self(batch)
            loss = self.criterion(outputs, labels)
            self.log("test_loss", loss)

            return loss

        def configure_optimizers(self):
            optimizer = AdamW(self.parameters(), lr=self.lr)
            #         warmup_steps = self.steps_per_epoch//3
            #         total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

            scheduler = get_linear_schedule_with_warmup(optimizer, 500, 10000)

            return [optimizer], [scheduler]

    data = DataModule(
        train_df,
        val_df,
        test_df,
        tokenizer,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    data.setup()
    model = DistilClassifier(n_classes=args.n_classes)
    trainer = pl.Trainer(
        gpus=1,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        default_root_dir="/tmp/",
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(model, data)

    # Output
    predictions = trainer.predict(model, dataloaders=data.test_dataloader())
    predictions = torch.cat(predictions).cpu()
    with open(Path(args.output_data_dir) / "test_predictions.pickle", "wb") as f:
        pickle.dump(predictions, f)

    tokenizer.save_pretrained(Path(args.model_dir) / "tokenizer")
    trainer.save_checkpoint(Path(args.model_dir) / "model.ckpt")
