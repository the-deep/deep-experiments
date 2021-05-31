from pathlib import Path
import os
import random
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import torch
import sagemaker
from sagemaker import get_execution_role
import boto3
import pytorch_lightning as pl
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--model_name", type=str)

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

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "stupid_metric": 1.0,
        }

    # download model from model hub
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)


    class TransformerDataset(Dataset):
        def __init__(self, tokenizer, df):
            self.tokenizer = tokenizer
            self.labels = df['labels']
            self.texts = df['texts']

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            inputs = self.tokenizer.encode_plus(
                text,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            inputs = {
                'input_ids': inputs['input_ids'].flatten(),
                'attention_mask': inputs['attention_mask'].flatten(),
                'label': torch.tensor(self.labels[idx], dtype=torch.float)
            }
            return inputs


    class DataModule(pl.LightningDataModule):

        def __init__(self, train_df, val_df, test_df, tokenizer, batch_size=16, max_token_len=200):
            super().__init__()
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.max_token_len = max_token_len

        def setup(self):
            self.train_dataset = TransformerDataset(tokenizer=self.tokenizer, df=self.train_df)
            self.val_dataset = TransformerDataset(tokenizer=self.tokenizer, df=self.val_df)
            self.test_dataset = TransformerDataset(tokenizer=self.tokenizer, df=self.test_df)

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size=self.batch_size)

        def test_dataloader(self):
            return DataLoader(self.test_dataset, batch_size=self.batch_size)

    # we will use the BERT base model(the smaller one)
    class DistilClassifier(pl.LightningModule):
        # Set up the classifier
        def __init__(self, n_classes=10, steps_per_epoch=None, n_epochs=3, lr=2e-5):
            super().__init__()

            self.model = AutoModel.from_pretrained(model_name)
            self.classifier = nn.Linear(self.model.config.hidden_size,
                                        n_classes)
            self.steps_per_epoch = steps_per_epoch
            self.n_epochs = n_epochs
            self.lr = lr
            self.criterion = nn.BCEWithLogitsLoss()

        def forward(self, input_ids, attn_mask):
            output = self.model(input_ids=input_ids, attention_mask=attn_mask)
            pooler_output = output.last_hidden_state[:, 0, :]
            output = self.classifier(pooler_output)

            return output

        def training_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            outputs = self(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            self.log('train_loss', loss, prog_bar=True, logger=True)

            return {"loss": loss, "predictions": outputs, "labels": labels}

        def validation_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            outputs = self(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            self.log('val_loss', loss, prog_bar=True, logger=True)

            return loss

        def test_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            outputs = self(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            self.log('test_loss', loss, prog_bar=True, logger=True)

            return loss

        def configure_optimizers(self):
            optimizer = AdamW(self.parameters(), lr=self.lr)
            #         warmup_steps = self.steps_per_epoch//3
            #         total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

            scheduler = get_linear_schedule_with_warmup(optimizer, 500, 10000)

            return [optimizer], [scheduler]


    data = DataModule(train_df, val_df, test_df, tokenizer)
    data.setup()
    model = DistilClassifier()
    trainer = pl.Trainer()
    trainer.fit(model, data)
