import logging
import sys
import argparse
import os
from pathlib import Path
import pickle

from transformers import (
    DistilBertForMaskedLM,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--text_column_mlm", type=str)
    parser.add_argument("--label_column", type=str)

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
    # train_dataset = load_from_disk(args.training_dir)
    train_df = pd.read_pickle(f"{args.training_dir}/train_df.pickle")
    val_df = pd.read_pickle(f"{args.training_dir}/val_df.pickle")
    test_df = pd.read_pickle(f"{args.training_dir}/test_df.pickle")
    logger.info(f" loaded train_dataset length is: {train_df.shape}")
    logger.info(f" loaded val_dataset length is: {val_df.shape}")
    logger.info(f" loaded test_dataset length is: {test_df.shape}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    model = DistilBertForMaskedLM.from_pretrained(args.model_name)

    class LMBFFDataset(Dataset):
        def __init__(self, tokenizer, df, text_col, label_col):
            self.tokenizer = tokenizer
            self.df = df
            self.texts = df[text_col].values
            self.labels = self.compute_labels(df[label_col])

        def compute_labels(self, labels):
            yes_token = self.tokenizer.convert_tokens_to_ids("yes")
            no_token = self.tokenizer.convert_tokens_to_ids("no")
            # labels = [[yes_token if y else no_token for y in label] for label in labels]
            labels = [yes_token if label else no_token for label in labels]
            return torch.tensor(labels)

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, idx):
            text = self.texts[idx]
            inputs = self.tokenizer.encode_plus(
                text,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            inputs["input_ids"] = inputs["input_ids"].squeeze(0)
            labels = self.labels[idx]
            inputs["labels"] = self.mlm_labels(inputs, labels)
            return inputs

        def mlm_labels(self, inputs, label):
            labels = inputs["input_ids"].clone()
            labels[labels != tokenizer.mask_token_id] = -100
            labels[labels == tokenizer.mask_token_id] = label
            return labels

    train_dataset = LMBFFDataset(tokenizer, train_df, args.text_column_mlm, args.label_column)
    val_dataset = LMBFFDataset(tokenizer, val_df, args.text_column_mlm, args.label_column)
    test_dataset = LMBFFDataset(tokenizer, test_df, args.text_column_mlm, args.label_column)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.predict(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    output_path = Path(args.output_data_dir)
    with open(str(output_path / "prediction.pickle"), "wb") as f:
        pickle.dump(eval_result, f)

    # Saves the model to s3
    trainer.save_model(args.model_dir)
