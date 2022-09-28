import sys

sys.path.append(".")

import math
import pickle
import logging
from typing import List
from pathlib import Path
from dataclasses import dataclass, field
import os
import argparse
import torch
import mlflow
import pytorch_lightning as pl
from transformers import (
    TrainingArguments,
)
from transformers.hf_argparser import HfArgumentParser

from model import TrainingExtractionModel, LoggedExtractionModel
from data import (
    LABEL_NAMES,
)
from inference import EntryExtractionWrapper

logging.basicConfig(level=logging.INFO)

MLFLOW_SERVER = "http://mlflow-deep-387470f3-1883319727.us-east-1.elb.amazonaws.com/"
SAGEMAKER_ROLE = "AmazonSageMaker-ExecutionRole-20210519T102514"


@dataclass
class Args:
    model_name_or_path: str
    data_path: str
    excerpts_csv_path: str
    max_full_length: int
    max_length: int
    extra_context_length: int
    tracking_uri: str = MLFLOW_SERVER
    experiment_name: str = "entry_extraction"
    n_separate_layers: int = 0
    separate_layer_groups: List[List[str]] = list
    token_loss_weight: float = 1.0
    sentence_edit_threshold: int = math.inf
    n_subsample: int = None
    compute_relevant_with_or: bool = False
    loss_weights: List[float] = field(
        default_factory=lambda: [1.0] + [0.0] * (len(LABEL_NAMES) - 1)
    )


def get_conda_env_specs():
    requirement_file = str(Path(__file__).parent / "requirements.txt")
    with open(requirement_file, "r") as f:
        requirements = f.readlines()
    requirements = [x.replace("\n", "") for x in requirements]

    default_env = mlflow.pytorch.get_default_conda_env()
    pip_dependencies = default_env["dependencies"][2]["pip"]
    pip_dependencies.extend(requirements)
    return default_env


def get_args():
    parser = HfArgumentParser([Args, TrainingArguments])

    (args, training_args) = parser.parse_json_file(sys.argv[1])
    return args, training_args


def get_separate_layer_groups(args):
    if args.separate_layer_groups is not None:
        separate_layer_groups = []

        for group in args.separate_layer_groups:
            separate_layer_groups.append(
                [LABEL_NAMES.index(label_name) for label_name in group]
            )
    else:
        separate_layer_groups = args.separate_layer_groups
    return separate_layer_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--tracking_uri", type=str)

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    # parser.add_argument("--val_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    default_args, _ = parser.parse_known_args()

    logging.info("building training and testing datasets")

    args, training_args = get_args()

    # load data
    with open(default_args.data_dir, "rb") as f:
        train_dataset, test_dataset, val_dataset = pickle.load(f)

    # Set remote mlflow server
    mlflow.set_tracking_uri(args.tracking_uri)
    # Set experiment name
    mlflow.set_experiment(args.experiment_name)
    # pytorch autolog automatically logs relevant data. DO NOT log the model, since
    # for NLP tasks we need a custom inference logic
    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run():

        train_params = {"batch_size": 256, "shuffle": True, "num_workers": 0}
        val_params = {"batch_size": 256, "shuffle": False, "num_workers": 0}

        params = {
            "train": train_params,
            "val": val_params,
        }
        mlflow.log_params(params)  # Logging example

        logging.info("training model")

        if torch.cuda.is_available():
            gpu_nb = 1
            training_device = "cuda"
        else:
            gpu_nb = 0
            training_device = "cpu"

        ##################### train model using pytorch lightning #####################

        # to be changed later
        early_stopping_callback = None
        checkpoint_callback = None

        trainer = pl.Trainer(
            logger=None,
            # callbacks=[early_stopping_callback, checkpoint_callback],
            progress_bar_refresh_rate=20,
            profiler="simple",
            # log_gpu_memory=True,
            weights_summary=None,
            gpus=gpu_nb,
            # precision=16,
            accumulate_grad_batches=1,
            max_epochs=training_args.num_train_epochs,
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
            # strategy="deepspeed_stage_3"
            # overfit_batches=1,
            # limit_predict_batches=2,
            # limit_test_batches=2,
            # fast_dev_run=True,
            # limit_train_batches=1,
            # limit_val_batches=1,
            # limit_test_batches: Union[int, float] = 1.0,
        )

        training_model = TrainingExtractionModel(
            args.model_name_or_path,
            num_labels=len(LABEL_NAMES),
            token_loss_weight=args.token_loss_weight,
            loss_weights=args.loss_weights,
            slice_length=args.max_length,
            extra_context_length=args.extra_context_length,
            n_separate_layers=args.n_separate_layers,
            separate_layer_groups=get_separate_layer_groups(args),
        )

        training_loader = training_model._get_loaders(train_dataset, train_params)
        val_loader = training_model._get_loaders(val_dataset, **val_params)

        trainer.fit(training_model, training_loader, val_loader)

        ###################### new model, used for logging, torch.nn.Module type #####################
        ###################### avoids logging errors #####################

        logged_extraction_model = LoggedExtractionModel(training_model)

        # This class is logged as a pickle artifact and used at inference time
        prediction_wrapper = EntryExtractionWrapper(logged_extraction_model)
        mlflow.pyfunc.log_model(
            python_model=prediction_wrapper,
            artifact_path="model",
            conda_env=get_conda_env_specs(),  # python conda dependencies
            code_path=[
                __file__,
                "model.py",
                "data.py",
                "inference.py",
                "utils.py",
                "config.json",
                "requirements.txt",
                "datasets.pkl",
            ],  # file dependencies
        )
