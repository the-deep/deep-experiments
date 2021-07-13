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
from torch.utils.data import DataLoader
import pandas as pd

from model import SectorsTransformer
from data import SectorsDataset
from inference import TransformersQAWrapper

logging.basicConfig(level=logging.INFO)


def get_conda_env_specs():
    requirement_file = str(Path(__file__).parent / "requirements.txt")
    with open(requirement_file, "r") as f:
        requirements = f.readlines()
    requirements = [x.replace("\n", "") for x in requirements]

    default_env = mlflow.pytorch.get_default_conda_env()
    pip_dependencies = default_env["dependencies"][2]["pip"]
    pip_dependencies.extend(requirements)
    return default_env


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

    # Set remote mlflow server
    mlflow.set_tracking_uri(args.tracking_uri)
    # Set experiment name
    mlflow.set_experiment(args.experiment_name)
    # pytorch autolog automatically logs relevant data. DO NOT log the model, since
    # for NLP tasks we need a custom inference logic
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
        mlflow.log_params(params)  # Logging example

        logging.info("training model")

        trainer = pl.Trainer(gpus=1, max_epochs=args.epochs)
        model = SectorsTransformer(args.model_name, len(class_to_id))
        trainer.fit(model, training_loader, val_loader)

        # This class is logged as a pickle artifact and used at inference time
        prediction_wrapper = TransformersQAWrapper(tokenizer, model)
        mlflow.pyfunc.log_model(
            python_model=prediction_wrapper,
            artifact_path="model",
            conda_env=get_conda_env_specs(),  # python conda dependencies
            code_path=[__file__, "model.py", "data.py", "inference.py"],  # file dependencies
        )
