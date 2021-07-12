# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import sys

sys.path.append(".")

import os
import logging
import argparse
from pathlib import Path

import mlflow
import pytorch_lightning as pl
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import pandas as pd

from model import SectorsTransformer
from data import SectorsDataset

logging.basicConfig(level=logging.INFO)


# class SectorsDatasetPreds(Dataset):
#     def __init__(self, dataframe, tokenizer, max_len=200):
#         self.tokenizer = tokenizer
#         self.excerpt_text = dataframe["excerpt"].tolist() if dataframe is not None else None
#         self.max_len = max_len
#
#     def encode_example(self, excerpt_text: str):
#         inputs = self.tokenizer(
#             excerpt_text,
#             None,
#             truncation=True,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding="max_length",
#             return_token_type_ids=True,
#         )
#         ids = inputs["input_ids"]
#         mask = inputs["attention_mask"]
#         token_type_ids = inputs["token_type_ids"]
#         encoded = {
#             "ids": torch.tensor(ids, dtype=torch.long),
#             "mask": torch.tensor(mask, dtype=torch.long),
#             "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
#         }
#         return encoded
#
#     def __len__(self):
#         return len(self.excerpt_text)
#
#     def __getitem__(self, index):
#         excerpt_text = str(self.excerpt_text[index])
#         return self.encode_example(excerpt_text)


class TransformersQAWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model.eval()
        super().__init__()

    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        dataset = SectorsDataset(model_input, self.tokenizer)
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
    training_set = SectorsDataset(
        dataframe=train_df, tokenizer=tokenizer, class_to_id=class_to_id, max_len=args.max_len
    )
    val_set = SectorsDataset(
        dataframe=val_df, tokenizer=tokenizer, class_to_id=class_to_id, max_len=args.max_len
    )

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
        logging.info(__file__)
        mlflow.pyfunc.log_model(
            python_model=prediction_wrapper,
            artifact_path="model",
            conda_env=default_env,
            code_path=[__file__, "model.py", "data.py"],
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
