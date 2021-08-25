import os
import argparse
import logging

import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
import timeit

import mlflow

from utils import (
    read_merge_data,
    preprocess_data,
    tagname_to_id,
    compute_weights,
)
from inference import TransformersQAWrapper
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
    parser.add_argument("--training_column", type=str, default="subpillars")

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

    # load datasets
    logging.info("reading, preprocessing data")

    all_dataset = read_merge_data(
        args.training_dir, args.val_dir, training_column=args.training_column, data_format="pickle"
    )

    train_df, val_df = preprocess_data(
        all_dataset, perform_augmentation=False, method=args.method_language
    )

    if args.train_with_whole_dataset:
        train_df = pd.concat([train_df, val_df])

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

        tags_ids = tagname_to_id(all_dataset.target)
        list_tags = list(tags_ids.keys())

        number_data_classes = []
        for tag in list_tags:
            nb_data_in_class = train_df.target.apply(lambda x: tag in (x)).sum()
            number_data_classes.append(nb_data_in_class)

        weights = compute_weights(number_data_classes, train_df.shape[0])

        weights = [weight if weight < 5 else weight ** 2 for weight in weights]

        log_dir_name = "-".join(args.model_name.split("/"))
        PATH_NAME = args.model_dir
        if not os.path.exists(PATH_NAME):
            os.makedirs(PATH_NAME)

        early_stopping_callback = EarlyStopping(monitor="val_f1", patience=2, mode="max")

        checkpoint_callback_params = {
            "save_top_k": 1,
            "verbose": True,
            "monitor": "val_f1",
            "mode": "max",
        }

        FILENAME = "model_" + args.training_column
        dirpath_pillars = str(PATH_NAME)
        checkpoint_callback_pillars = ModelCheckpoint(
            dirpath=dirpath_pillars, filename=FILENAME, **checkpoint_callback_params
        )

        model_subpillars = train_on_specific_targets(
            train_df,
            val_df,
            FILENAME,
            args.model_name,
            args.tokenizer_name,
            early_stopping_callback,
            checkpoint_callback_pillars,
            gpu_nb=1,
            train_params=train_params,
            val_params=val_params,
            MAX_EPOCHS=args.epochs,
            dropout_rate=args.dropout_rate,
            weight_classes=weights,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            max_len=args.max_len,
            warmup_steps=args.warmup_steps,
            pred_threshold=float(args.pred_threshold),
            output_length=args.output_length,
            multiclass_bool=args.multiclass_bool,
        )

        # This class is logged as a pickle artifact and used at inference time
        prediction_wrapper = TransformersQAWrapper(model_subpillars)
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
        (
            metrics_pillars,
            metrics_subpillars,
            ratio_evaluated_sentences,
        ) = model_subpillars.custom_eval(val_df)
        stop = timeit.default_timer()

        sentences_per_second = val_df.shape[0] / (stop - start)
        mlflow.log_metric("nb sentences per second", sentences_per_second)
        mlflow.log_metric("ratio of evaluated sentences", ratio_evaluated_sentences)

        if metrics_pillars.equals(metrics_subpillars):
            mlflow.log_metric("average macro f1 score", metrics_subpillars.loc["mean"]["F1 Score"])
            mlflow.log_metric("average accuracy", metrics_subpillars.loc["mean"]["Accuracy"])
            mlflow.log_metric("average precision", metrics_subpillars.loc["mean"]["Precision"])
            mlflow.log_metric("average recall", metrics_subpillars.loc["mean"]["Recall"])

            list_names = list(metrics_subpillars["Sector"])

            for i in range(len(list_names) - 1):
                try:
                    name = list_names[i]
                    cleaned_name = ''.join(x for x in name if x.isalpha())
                    final_name = "f1 score " + cleaned_name
                    
                    mlflow.log_metric(final_name, metrics_subpillars.iloc[i]["F1 Score"])
                except Exception:
                    pass

        else:
            mlflow.log_metric("supillar f1 score", metrics_subpillars.loc["mean"]["F1 Score"])
            mlflow.log_metric("supillar accuracy", metrics_subpillars.loc["mean"]["Accuracy"])
            mlflow.log_metric("supillar precision", metrics_subpillars.loc["mean"]["Precision"])
            mlflow.log_metric("supillar recall", metrics_subpillars.loc["mean"]["Recall"])

            mlflow.log_metric("pillar f1 score", metrics_pillars.loc["mean"]["F1 Score"])
            mlflow.log_metric("pillar accuracy", metrics_pillars.loc["mean"]["Accuracy"])
            mlflow.log_metric("pillar precision", metrics_pillars.loc["mean"]["Precision"])
            mlflow.log_metric("pillar recall", metrics_pillars.loc["mean"]["Recall"])
