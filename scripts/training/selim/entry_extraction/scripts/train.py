import sys

sys.path.append(".")

import logging
from typing import List
from pathlib import Path
import os
import pandas as pd
import argparse
import torch
import mlflow
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from ast import literal_eval
from model import TrainingExtractionModel, LoggedExtractionModel
from inference import EntryExtractionWrapper

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

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--tokenizer_name_or_path", type=str)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--extra_context_length", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--dataloader_num_workers", type=int, default=6)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--instance_type", type=str)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fbeta", type=float, default=2)

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
    args, _ = parser.parse_known_args()

    logging.info("building training and testing datasets")

    # load data
    full_data = pd.read_pickle(f"{args.data_dir}/data.pickle")

    train_dataset = literal_eval(
        full_data[full_data.col_name == "train"].vals.values[0]
    )
    test_dataset = literal_eval(full_data[full_data.col_name == "test"].vals.values[0])
    val_dataset = literal_eval(full_data[full_data.col_name == "val"].vals.values[0])
    tagname_to_tagid = literal_eval(
        full_data[full_data.col_name == "tagname_to_id"].vals.values[0]
    )

    # Set remote mlflow server
    mlflow.set_tracking_uri(args.tracking_uri)
    # Set experiment name
    mlflow.set_experiment(args.experiment_name)
    # pytorch autolog automatically logs relevant data. DO NOT log the model, since
    # for NLP tasks we need a custom inference logic
    mlflow.pytorch.autolog(log_models=False)

    if torch.cuda.is_available():
        gpu_nb = 1
        training_device = "cuda"
    else:
        gpu_nb = 0
        training_device = "cpu"

    with mlflow.start_run():

        train_params = {
            "batch_size": args.train_batch_size,
            "shuffle": True,
            "num_workers": args.dataloader_num_workers,
        }
        val_params = {
            "batch_size": args.val_batch_size,
            "shuffle": False,
            "num_workers": args.dataloader_num_workers,
        }

        data_params = {
            "train": train_params,
            "val": val_params,
        }
        mlflow.log_params(data_params)

        model_params = {
            "epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "model_name": args.model_name_or_path,
            "tokenizer_name": args.tokenizer_name_or_path,
            "instance_type": args.instance_type,
            "n_gpu": gpu_nb,
            "max_len": args.max_len,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "fbeta": args.fbeta,
        }

        mlflow.log_params(model_params)

        logging.info("training model")

        ##################### train model using pytorch lightning #####################

        MODEL_NAME = "transformer_model"
        MODEL_DIR = args.model_dir
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=1 + args.n_epochs // 3,
            mode="min",
        )

        checkpoint_callback_params = {
            "save_top_k": 1,
            "verbose": True,
            "monitor": "val_loss",
            "mode": "min",
        }

        checkpoint_callback = ModelCheckpoint(
            dirpath=MODEL_DIR, filename=MODEL_NAME, **checkpoint_callback_params
        )

        trainer = pl.Trainer(
            logger=None,
            callbacks=[early_stopping_callback, checkpoint_callback],
            # progress_bar_refresh_rate=20,
            profiler="simple",
            # log_gpu_memory=True,
            # weights_summary=None,
            gpus=gpu_nb,
            # precision=16,
            accumulate_grad_batches=1,
            max_epochs=args.n_epochs,
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
            backbone_name=args.model_name_or_path,
            tokenizer_name=args.tokenizer_name_or_path,
            tagname_to_tagid=tagname_to_tagid,
            slice_length=args.max_len,
            extra_context_length=args.extra_context_length,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        training_loader = training_model._get_loaders(
            train_dataset, train_params, training_mode=True
        )
        val_loader = training_model._get_loaders(
            val_dataset, val_params, training_mode=True
        )

        trainer.fit(training_model, training_loader, val_loader)

        ###################### new model, used for logging, torch.nn.Module type #####################
        ###################### avoids logging errors #####################
        training_model.eval()
        training_model.freeze()

        logged_extraction_model = LoggedExtractionModel(training_model)

        val_results = logged_extraction_model.hypertune_threshold(
            val_loader, args.fbeta
        )

        # log tag results
        for tag_name, tag_results in val_results.items():
            mlflow.log_metrics(tag_results)

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
                "requirements.txt",
            ],  # file dependencies
        )
