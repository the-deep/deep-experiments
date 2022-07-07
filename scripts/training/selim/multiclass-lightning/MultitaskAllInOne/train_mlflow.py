import os

# setting tokenizers parallelism to false adds robustness when dploying the model
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dill import needs to be kept for more robustness in multimodel serialization
import dill

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

dill.extend(True)

import multiprocessing

import argparse
import logging
from ast import literal_eval

from pathlib import Path

import mlflow

from utils import read_merge_data, preprocess_df, clean_name_for_logging
from model import train_model
import torch

from inference import TransformersPredictionsWrapper

import torch

import numpy as np


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--tracking_uri", type=str)

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)

    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--output_length", type=int, default=384)
    parser.add_argument("--nb_repetitions", type=int, default=1)
    parser.add_argument(
        "--model_name", type=str, default="microsoft/xtremedistil-l6-h384-uncased"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default="microsoft/xtremedistil-l6-h384-uncased"
    )
    # parser.add_argument("--log_every_n_steps", type=int, default=10)
    # parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--training_names", type=str)
    parser.add_argument("--f_beta", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--min_results", type=str)
    parser.add_argument("--run_name", type=str, default="models")
    parser.add_argument("--instance_type", type=str, default="-")
    parser.add_argument("--only_backpropagate_pos", type=str, default="false")

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    parser.add_argument(
        "--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument("--val_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    args, _ = parser.parse_known_args()

    def get_conda_env_specs():
        requirement_file = str(Path(__file__).parent / "requirements.txt")
        with open(requirement_file, "r") as f:
            requirements = f.readlines()
        requirements = [x.replace("\n", "") for x in requirements]

        default_env = mlflow.pytorch.get_default_conda_env()
        pip_dependencies = default_env["dependencies"][2]["pip"]
        pip_dependencies.extend(requirements)
        return default_env

    # load datasets
    logging.info("reading, preprocessing data")

    whole_df, test_df = read_merge_data(
        args.training_dir, args.val_dir, data_format="pickle"
    )

    # Set remote mlflow server
    mlflow.set_tracking_uri(args.tracking_uri)
    # Set experiment name
    mlflow.set_experiment(args.experiment_name)
    # pytorch autolog automatically logs relevant data. DO NOT log the model, since
    # for NLP tasks we need a custom inference logic
    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run():
        train_params = {
            "batch_size": args.train_batch_size,
            "shuffle": True,
            "num_workers": 2,
        }
        val_params = {
            "batch_size": args.val_batch_size,
            "shuffle": False,
            "num_workers": 2,
        }

        if torch.cuda.is_available():
            gpu_nb = 1
            training_device = "cuda"
        else:
            gpu_nb = 0
            training_device = "cpu"

        only_backpropagate_pos = args.only_backpropagate_pos != "false"

        params = {
            "epochs": args.epochs,
            "warmup_steps": args.warmup_steps,
            "learning_rate": args.learning_rate,
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "f_beta": args.f_beta,
            "instance_type": args.instance_type,
            "n_gpu": gpu_nb,
            "only_backpropagate_pos": only_backpropagate_pos,
            "max_len": args.max_len,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay
        }

        mlflow.log_params(params)
        mlflow.log_param("train_batch_size", args.train_batch_size)

        train_df, val_df = preprocess_df(whole_df)


        MODEL_NAME = "classification_model"

        model = train_model(
            train_dataset=train_df,
            val_dataset=val_df,
            MODEL_DIR=args.model_dir,
            MODEL_NAME=MODEL_NAME,
            BACKBONE_NAME=args.model_name,
            TOKENIZER_NAME=args.tokenizer_name,
            gpu_nb=gpu_nb,
            train_params=train_params,
            val_params=val_params,
            MAX_EPOCHS=args.epochs,
            dropout_rate=args.dropout,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            output_length=args.output_length,
            multiclass_bool=True,
            training_device=training_device,
            f_beta=args.f_beta,
            only_backpropagate_pos=only_backpropagate_pos,
            max_len=args.max_len,
        )

        try:
            model_path = str(Path(args.model_dir) / MODEL_NAME)
            mlflow.log_param("model_path", model_path)
            """torch.save(model.state_dict(), model_path)"""
        except Exception as e:
            logging.info("model_state_dict_error", e)

        for metric, scores in model.optimal_scores.items():
            mlflow.log_metrics(clean_name_for_logging(scores, context=metric))
            mlflow.log_metric(f"aa_mean_{metric}", np.mean(list(scores.values())))

        mlflow.log_metrics(
            clean_name_for_logging(model.optimal_thresholds, context="thresholds")
        )

        pyfunc_prediction_wrapper = TransformersPredictionsWrapper(model)

        try:
            mlflow.pyfunc.log_model(
                python_model=pyfunc_prediction_wrapper,
                artifact_path=args.run_name,
                conda_env=get_conda_env_specs(),  # python conda dependencies
                code_path=[
                    __file__,
                    "model.py",
                    "utils.py",
                    "inference.py",
                    "data.py",
                    "pooling.py",
                    "architecture.py",
                ],
                await_registration_for=600,
            )
        except Exception as e:
            logging.info("pyfunc", e)

    # testing
    """test_excerpts = test_df["excerpt"]

    predictions_test_set = model.custom_predict(test_excerpts, testing=True)

    outputs = {
        "preds_test_set": predictions_test_set,
        "thresholds": model.optimal_thresholds,
    }

    with open(Path(args.output_data_dir) / "logged_values.pickle", "wb") as f:
        dill.dump(outputs, f)"""
