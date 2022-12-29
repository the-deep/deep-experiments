import os

# setting tokenizers parallelism to false adds robustness when dploying the model
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dill import needs to be kept for more robustness in multimodel serialization
import dill

dill.extend(True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import multiprocessing
import argparse
import time

from pathlib import Path
import mlflow
import copy
import torch
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict
from utils import (
    preprocess_df,
    clean_name_for_logging,
    get_n_tokens,
    create_train_val_df,
    _get_labled_unlabled_data,
    _create_stratified_train_test_df,
    _create_df_with_translations,
)


from ModelTraining import train_model
from Inference import ClassificationInference

import logging

logging.basicConfig(level=logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--tracking_uri", type=str)

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)

    parser.add_argument("--max_len", type=int, default=512)
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
    parser.add_argument("--run_name", type=str, default="models")
    parser.add_argument("--instance_type", type=str, default="-")
    parser.add_argument("--n_freezed_layers", type=int, default=1)
    parser.add_argument("--predictions_on_test_set", type=str, default="true")
    parser.add_argument("--explainability", type=str, default="true")
    parser.add_argument("--delete_long_excerpts", type=str, default="true")
    parser.add_argument("--loss_gamma", type=float, default=2)
    parser.add_argument("--proportions_pow", type=float, default=1)
    parser.add_argument(
        "--min_entries_per_proj", type=int
    )  # TODO: add args in notebook

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

    # Set remote mlflow server
    mlflow.set_tracking_uri(args.tracking_uri)
    # Set experiment name
    mlflow.set_experiment(args.experiment_name)
    # pytorch autolog automatically logs relevant data. DO NOT log the model, since
    # for NLP tasks we need a custom inference logic
    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run():
        train_params_transformer = {
            "batch_size": args.train_batch_size,
            "shuffle": True,
            "num_workers": 4,
        }
        val_params_transformer = {
            "batch_size": args.val_batch_size,
            "shuffle": False,
            "num_workers": 4,
        }

        if torch.cuda.is_available():
            gpu_nb = 1
            training_device = "cuda"
        else:
            gpu_nb = 0
            training_device = "cpu"

        params = {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "f_beta": args.f_beta,
            "instance_type": args.instance_type,
            "n_gpu": gpu_nb,
            "max_len": args.max_len,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "n_freezed_layers": args.n_freezed_layers,
            "min_entries_per_proj": args.min_entries_per_proj,
        }

        mlflow.log_params(params)
        mlflow.log_param("train_batch_size", args.train_batch_size)

        ###########################     Data Preparation     ##############################

        # pull data
        all_data = pd.read_pickle(f"{args.training_dir}/train.pickle")
        all_data, projects_list_per_tag, grouped_tags = preprocess_df(
            all_data, args.min_entries_per_proj
        )

        # Stratified splitting project-wise
        train_val_df, test_df = _create_stratified_train_test_df(all_data)

        if args.delete_long_excerpts == "true":
            n_tokens = get_n_tokens(train_val_df.excerpt.tolist())
            train_val_df = train_val_df.iloc[n_tokens <= int(args.max_len * 1.5)]

        # final format for training
        train_val_df = _create_df_with_translations(train_val_df)

        ###############################     Apply Relabling     ###############################

        # Dict[int: entry_id, List[str]: final tags]
        train_val_final_labels = defaultdict(list)

        # TODO: Cross relabling: treat sectors first hen others
        ...

        for tags_with_same_projects in grouped_tags:
            projects_list_one_same_tags_set = projects_list_per_tag[
                tags_with_same_projects[0]
            ]

            (
                train_val_data_labeled,
                train_val_data_one_tag_non_labeled,
                test_data_one_tag_labeled,
            ) = _get_labled_unlabled_data(
                train_val_df,
                test_df,
                projects_list_one_same_tags_set,
                tags_with_same_projects,
            )

            # classification model name
            classification_transformer_name = (
                f"model_{clean_name_for_logging('_'.join(tags_with_same_projects))}"
            )

            train_df_one_tag, val_df_one_tag = create_train_val_df(
                train_val_data_labeled
            )

            transformer_model = train_model(
                train_dataset=train_df_one_tag,
                val_dataset=val_df_one_tag,
                MODEL_DIR=args.model_dir,
                MODEL_NAME=classification_transformer_name,
                BACKBONE_NAME=args.model_name,
                TOKENIZER_NAME=args.tokenizer_name,
                gpu_nb=gpu_nb,
                loss_gamma=args.loss_gamma,
                proportions_pow=args.proportions_pow,
                train_params=train_params_transformer,
                val_params=val_params_transformer,
                MAX_EPOCHS=args.epochs,
                dropout_rate=args.dropout,
                weight_decay=args.weight_decay,
                learning_rate=args.learning_rate,
                output_length=args.output_length,
                training_device=training_device,
                f_beta=args.f_beta,
                max_len=args.max_len,
                n_freezed_layers=args.n_freezed_layers,
                hypertune_threshold_bool=True,
            )

            # generate predictions on labeled test set
            predictions_one_tag_test_data = transformer_model.custom_predict(
                test_data_one_tag_labeled.excerpt,
                return_transformer_only=False,
                hypertuning_threshold=False,
            )

            # TODO: results on test set
            ...

            # predictions on unlabeled train val df
            predictions_one_tag_unlabeled_excerpts = transformer_model.custom_predict(
                train_val_data_one_tag_non_labeled.excerpt,  # TODO: excerpt here or languages mean or what
                return_transformer_only=False,
                hypertuning_threshold=False,
            )

            # TODO: final results unlabeled df + one final results per entry_id
            ...

            # TODO: postprocessing: not including one tag if parent not there + take care of specific case of demographic groups
            ...

            final_predictions_unlabled_train_val = ...

            # update with groundtruths labels
            for i, row in train_val_data_labeled.iterrows():
                train_val_final_labels[row["entry_id"]].extend(row["target"])

            # update with newly labeled data
            for i in range(len(final_predictions_unlabled_train_val)):
                train_val_final_labels[
                    train_val_data_one_tag_non_labeled.iloc[i]["entry_id"]
                ].extend(final_predictions_unlabled_train_val[i])

        final_labels_df = pd.DataFrame(
            list(
                zip(
                    list(train_val_final_labels.keys()),
                    list(train_val_final_labels.values()),
                )
            ),
            columns=["entry_id", "target"],
        )

        # save predictions df
        final_labels_df.to_csv(
            Path(args.output_data_dir) / "final_labels_df.csv", index=None
        )

        train_val_df = pd.merge(
            left=train_val_df.drop(columns=["target"]),
            right=final_labels_df,
            on="entry_id",
        )

        ###############################  train backbone model   ###############################

        train_df, val_df = create_train_val_df(train_val_df)

        TRANSFORMER_MODEL_NAME = "all_tags_transformer_model"

        transformer_model = train_model(
            train_dataset=train_df,
            val_dataset=val_df,
            MODEL_DIR=args.model_dir,
            MODEL_NAME=TRANSFORMER_MODEL_NAME,
            BACKBONE_NAME=args.model_name,
            TOKENIZER_NAME=args.tokenizer_name,
            gpu_nb=gpu_nb,
            loss_gamma=args.loss_gamma,
            proportions_pow=args.proportions_pow,
            train_params=train_params_transformer,
            val_params=val_params_transformer,
            MAX_EPOCHS=args.epochs,
            dropout_rate=args.dropout,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            output_length=args.output_length,
            training_device=training_device,
            f_beta=args.f_beta,
            max_len=args.max_len,
            n_freezed_layers=args.n_freezed_layers,
            hypertune_threshold_bool=False,
        )

        Transformer_model_path = str(Path(args.model_dir) / TRANSFORMER_MODEL_NAME)
        mlflow.log_param("transformer_model_path", Transformer_model_path)

        # a deepcopy of the returned model is needed in order not to have pickling issues during logging
        # logged_model = copy.deepcopy(transformer_model)

        ClassificationPredictor = ClassificationInference(transformer_model)

        mlflow.pyfunc.log_model(
            python_model=ClassificationPredictor,
            artifact_path=f"two_steps_models",
            conda_env=get_conda_env_specs(),  # python conda dependencies
            code_path=[
                __file__,
                "loss.py",
                "utils.py",
                "Inference.py",
                "TransformerModel.py",
                "pooling.py",
                "ModelsExplainability.py",
            ],
            await_registration_for=600,
        )
