import os

# setting tokenizers parallelism to false adds robustness when dploying the model
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dill import needs to be kept for more robustness in multimodel serialization
import dill

dill.extend(True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import multiprocessing
import argparse
import json
from pathlib import Path
import mlflow
from copy import copy
import torch
from collections import defaultdict
import pandas as pd
from utils import (
    preprocess_df,
    _clean_str_for_logging,
    _clean_results_for_logging,
    _clean_thresholds_for_logging,
    _create_stratified_train_test_df,
    generate_results,
    _get_sectors_non_sectors_grouped_tags,
    _generate_test_set_results,
    _get_results_df_from_dict,
)

from models_training import train_model, _relabel_sectors, _relabel_subsectors


from Inference import ClassificationInference

import logging

logging.basicConfig(level=logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)


def train_test(
    train_val_data_labeled: pd.DataFrame,
    test_data_one_tag_labeled: pd.DataFrame,
    classification_transformer_name,
):
    transformer_model = train_model(
        MODEL_NAME=classification_transformer_name,
        train_val_dataset=train_val_data_labeled,
        hypertune_threshold_bool=True,
        **model_args,
    )

    # Generate predictions and results on labeled test set
    test_set_results = generate_results(
        transformer_model,
        test_data_one_tag_labeled.excerpt.tolist(),
        test_data_one_tag_labeled["target"].tolist(),
    )

    mlflow.log_metrics(_clean_results_for_logging(test_set_results, prefix="test"))
    mlflow.log_metrics(
        _clean_thresholds_for_logging(transformer_model.optimal_thresholds)
    )

    return transformer_model, test_set_results


def _log_results():
    # log results
    mlflow.log_metrics(_clean_results_for_logging(final_results, prefix="test"))
    # log thresholds
    mlflow.log_metrics(
        _clean_thresholds_for_logging(transformer_model.optimal_thresholds)
    )

    # save results dict
    with open(Path(args.output_data_dir) / "test_set_results.json", "w") as fp:
        json.dump(final_results, fp)

    # get results df
    results_as_df = _get_results_df_from_dict(final_results, proportions)

    # save results df
    results_as_df.to_csv(
        Path(args.output_data_dir) / "test_set_results.csv", index=None
    )


def _log_models():
    # log results as mlflow artifacts
    mlflow.pyfunc.log_model(
        python_model=ClassificationInference(transformer_model),
        artifact_path=f"classification_model_v0_8_gpu",
        conda_env=get_conda_env_specs(),  # python conda dependencies
        code_path=[
            __file__,
            "loss.py",
            "utils.py",
            "Inference.py",
            "TransformerModel.py",
            "pooling.py",
            "ModelsExplainability.py",
            "data.py",
        ],
        await_registration_for=600,
    )

    mlflow.pyfunc.log_model(
        python_model=ClassificationInference(transformer_model.to("cpu")),
        artifact_path=f"classification_model_v0_8_cpu",
        conda_env=get_conda_env_specs(),  # python conda dependencies
        code_path=[
            __file__,
            "loss.py",
            "utils.py",
            "Inference.py",
            "TransformerModel.py",
            "pooling.py",
            "ModelsExplainability.py",
            "data.py",
        ],
        await_registration_for=600,
    )


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
    parser.add_argument("--min_entries_per_proj", type=int, default=300)
    parser.add_argument("--relabling_min_ratio", type=float, default=0.05)

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
            "_epochs": args.epochs,
            "_learning_rate": args.learning_rate,
            "_model_name": args.model_name,
            "_tokenizer_name": args.tokenizer_name,
            "_f_beta": args.f_beta,
            "_instance_type": args.instance_type,
            "_n_gpu": gpu_nb,
            "_max_len": args.max_len,
            "_dropout": args.dropout,
            "_weight_decay": args.weight_decay,
            "_n_freezed_layers": args.n_freezed_layers,
            "_min_entries_per_proj": args.min_entries_per_proj,
            "_relabling_min_ratio": args.relabling_min_ratio,
            "_loss_gamma": args.loss_gamma,
            "_proportions_pow": args.proportions_pow,
        }

        model_args = {
            "BACKBONE_NAME": args.model_name,
            "TOKENIZER_NAME": args.tokenizer_name,
            "f_beta": args.f_beta,
            "max_len": args.max_len,
            "n_freezed_layers": args.n_freezed_layers,
            "MAX_EPOCHS": args.epochs,
            "dropout_rate": args.dropout,
            "weight_decay": args.weight_decay,
            "learning_rate": args.learning_rate,
            "output_length": args.output_length,
            "loss_gamma": args.loss_gamma,
            "proportions_pow": args.proportions_pow,
            "gpu_nb": gpu_nb,
            "train_params": train_params_transformer,
            "val_params": val_params_transformer,
            "training_device": training_device,
            "delete_long_excerpts": args.delete_long_excerpts == "true",
        }

        mlflow.log_params(params)
        mlflow.log_param("train_batch_size", args.train_batch_size)

        ###########################     Data Preparation     ##############################

        # pull data
        all_data = pd.read_pickle(f"{args.training_dir}/train.pickle")
        all_data, projects_list_per_tag, grouped_tags = preprocess_df(
            all_data, args.min_entries_per_proj
        )  # TODO: change this, no grouped tags
        sector_groups, non_sector_groups = _get_sectors_non_sectors_grouped_tags(
            grouped_tags
        )

        # Stratified splitting project-wise
        train_val_df, test_df = _create_stratified_train_test_df(all_data)

        ###############################     Apply Relabling     ###############################

        train_val_df = _relabel_sectors(train_val_df, projects_list_per_tag, model_args)
        train_val_df = _relabel_subsectors(train_val_df, model_args)

        ###############################  train backbone model   ###############################

        TRANSFORMER_MODEL_NAME = "all_tags_transformer_model"

        transformer_model = train_model(
            MODEL_NAME=TRANSFORMER_MODEL_NAME,
            train_val_dataset=train_val_df,
            hypertune_threshold_bool=True,
            **model_args,
        )
        proportions = copy(transformer_model.tags_proportions)
        proportions = {
            tagname: proportions[tagid]
            for tagname, tagid in transformer_model.tagname_to_tagid.items()
        }

        Transformer_model_path = str(Path(args.model_dir) / TRANSFORMER_MODEL_NAME)
        mlflow.log_metrics(
            {
                f"__proportions_{_clean_str_for_logging(tagname)}": transformer_model.tags_proportions[
                    tagid
                ].item()
                for tagname, tagid in transformer_model.tagname_to_tagid.items()
            }
        )
        mlflow.log_param("transformer_model_path", Transformer_model_path)

        final_results = _generate_test_set_results(
            transformer_model, test_df, projects_list_per_tag
        )

        _log_results()
        _log_models()
