import os
import re
import argparse
import logging

import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
import timeit

import mlflow

from utils import (
    read_merge_data,
    preprocess_df
)

from inference import TransformersPredictionsWrapper
from generate_models import train_on_specific_targets

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--tracking_uri", type=str)

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)

    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--pred_threshold", type=float, default=0.4)
    parser.add_argument("--output_length", type=int, default=384)

    parser.add_argument("--model_name", type=str, default="microsoft/xtremedistil-l6-h384-uncased")
    parser.add_argument(
        "--tokenizer_name", type=str, default="microsoft/xtremedistil-l6-h384-uncased"
    )
    # parser.add_argument("--log_every_n_steps", type=int, default=10)
    # parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--method_language", type=str, default="keep all")
    parser.add_argument("--training_names", type=str, default="sectors,subpillars_2d,subpillars_1d")

    parser.add_argument("--train_with_whole_dataset", type=bool, default=False)
    parser.add_argument("--multiclass_bool", type=bool, default=True)
    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
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

    def log_metrics_MLflow(subpillars_df: pd.DataFrame, pillars_df: pd.DataFrame, ratio_eval_sentences, column_name: str):
        mlflow.log_metric(f"ratio of evaluated sentences {column_name}", ratio_eval_sentences)

        if subpillars_df.equals(pillars_df):
            mlflow.log_metric(f"average macro f1 score {column_name}", pillars_df.loc["mean"]["F1 Score"])
            mlflow.log_metric(f"average accuracy {column_name}", pillars_df.loc["mean"]["Accuracy"])
            mlflow.log_metric(f"average precision {column_name}", pillars_df.loc["mean"]["Precision"])
            mlflow.log_metric(f"average recall {column_name}", pillars_df.loc["mean"]["Recall"])

            list_names = list(pillars_df["Sector"])

            for i in range(len(list_names) - 1):
                try:
                    name = list_names[i]
                    cleaned_name = new_string = re.sub("[^0-9a-zA-Z]+", "_", name)
                    final_name = "f1_score_" + cleaned_name
                    
                    mlflow.log_metric(final_name, pillars_df.iloc[i]["F1 Score"])
                except Exception:
                    pass

        else:
            mlflow.log_metric(f"supillar f1 score {column_name}", subpillars_df.loc["mean"]["F1 Score"])
            mlflow.log_metric(f"supillar accuracy {column_name}", subpillars_df.loc["mean"]["Accuracy"])
            mlflow.log_metric(f"supillar precision {column_name}", subpillars_df.loc["mean"]["Precision"])
            mlflow.log_metric(f"supillar recall {column_name}", subpillars_df.loc["mean"]["Recall"])

            mlflow.log_metric(f"pillar f1 score {column_name}", pillars_df.loc["mean"]["F1 Score"])
            mlflow.log_metric(f"pillar accuracy {column_name}", pillars_df.loc["mean"]["Accuracy"])
            mlflow.log_metric(f"pillar precision {column_name}", pillars_df.loc["mean"]["Precision"])
            mlflow.log_metric(f"pillar recall {column_name}", pillars_df.loc["mean"]["Recall"])

    # load datasets
    logging.info("reading, preprocessing data")

    all_dataframe = read_merge_data(
        args.training_dir, args.val_dir, data_format="pickle"
    )

    training_columns = args.training_names.split(',')
    #train_df, val_df = preprocess_data(
    #    all_dataset, perform_augmentation=False, method=args.method_language
    #)

    

    # Set remote mlflow server
    mlflow.set_tracking_uri(args.tracking_uri)
    # Set experiment name
    mlflow.set_experiment(args.experiment_name)
    # pytorch autolog automatically logs relevant data. DO NOT log the model, since
    # for NLP tasks we need a custom inference logic
    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run():
        train_params = {"batch_size": args.train_batch_size, "shuffle": True, "num_workers": 4}

        val_params = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": 4}

        params = {
            "epochs": args.epochs,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout_rate,
            "threshold": args.pred_threshold,
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "languages_trained": args.method_language,
            "augmentation_performed": "no augmentation",
            "weighted_loss": "sqrt weights",
            "train with whole dataset":args.train_with_whole_dataset
        }

        mlflow.log_params(params)

        models = []

        for column_name in training_columns:
            train_df, val_df = preprocess_df(all_dataframe, column_name)

            if args.train_with_whole_dataset:
                train_df = pd.concat([train_df, val_df])

            model = train_on_specific_targets(
                train_dataset=train_df,
                val_dataset=val_df,
                MODEL_DIR=args.model_dir,
                MODEL_NAME=args.model_name,
                TOKENIZER_NAME=args.tokenizer_name,
                training_column=column_name,
                gpu_nb=1,
                train_params=train_params,
                val_params=val_params,
                MAX_EPOCHS=args.epochs,
                dropout_rate=args.dropout_rate,
                weight_decay=args.weight_decay,
                learning_rate=args.learning_rate,
                max_len=args.max_len,
                warmup_steps=args.warmup_steps,
                pred_threshold=float(args.pred_threshold),
                output_length=args.output_length,
                multiclass_bool=args.multiclass_bool,
            )
            models.append(model)

        # This class is logged as a pickle artifact and used at inference time
        prediction_wrapper = TransformersPredictionsWrapper(models[0])
        mlflow.pyfunc.log_model(
            python_model=prediction_wrapper,
            artifact_path="model",
            conda_env=get_conda_env_specs(),  # python conda dependencies
            code_path=[
                __file__,
                "data_and_model.py",
                "utils.py",
                "inference.py", 
            ],  # file dependencies
        )

        start = timeit.default_timer()

        pillars_tot = []
        subpillars_tot = []
        ratio_tot = []

        for one_model in models:
            (
                metrics_pillars,
                metrics_subpillars,
                ratio_evaluated_sentences,
            ) = one_model.custom_eval(val_df)

            log_metrics_MLflow(metrics_subpillars, 
                                metrics_pillars, 
                                ratio_evaluated_sentences, 
                                one_model.column_name)

        stop = timeit.default_timer()

        sentences_per_second = val_df.shape[0] / (stop - start)
        mlflow.log_metric("nb sentences per second to predict all tags", sentences_per_second)
