import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import multiprocessing
import argparse
import json
from pathlib import Path
import mlflow
from typing import Dict
from copy import copy
import torch
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    _preprocess_df,
    _clean_str_for_logging,
    _clean_results_for_logging,
    _clean_thresholds_for_logging,
    _generate_results,
    _generate_test_set_results,
    _get_results_df_from_dict,
    _get_bar_colour,
    _get_handles,
)

from models_training import train_model, _relabel_sectors, _relabel_subsectors


from Inference import ClassificationInference

import logging

logging.basicConfig(level=logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)


def _generate_visualization(
    results_df: pd.DataFrame, context_type: str, relabling_str: str
):
    tags = results_df["tag"]
    level_0_tags = list(set([item.split("->")[0] for item in tags]))

    handles = _get_handles()

    for one_level_0 in level_0_tags:
        results_df_one_level_0 = (
            results_df[results_df.tag.apply(lambda x: x.split("->")[0] == one_level_0)]
            .copy()
            .sort_values(by="f_score", ascending=False)
        )
        results_df_one_level_0["level_1"] = results_df_one_level_0["tag"].apply(
            lambda x: x.split("->")[1]
        )
        results_df_one_level_0["level_2"] = results_df_one_level_0["tag"].apply(
            lambda x: x.split("->")[2]
        )
        all_level_1 = list(set(list(results_df_one_level_0["level_1"])))

        fig, axes = plt.subplots(
            len(all_level_1), 1, sharex=True, figsize=(20, 14), facecolor="white"
        )

        ordered_level_1 = (
            results_df_one_level_0.copy()
            .groupby("level_1", as_index=False)
            .agg({"level_2": lambda x: len(list(x))})
            .sort_values(by="level_2", ascending=False)
            .level_1.tolist()
        )

        for i, one_level_1 in enumerate(ordered_level_1):
            level_2_tags_df = results_df_one_level_0[
                results_df_one_level_0.level_1 == one_level_1
            ]

            custom_palette = {}
            for _, row in level_2_tags_df.iterrows():
                score = row["f_score"]
                tagname = row["level_2"]
                custom_palette[tagname] = _get_bar_colour(score)

            axes[i].set_title(f"{one_level_1}", fontsize=14)
            # plt.gcf().autofmt_xdate()
            # axes[i].xaxis.set_visible(False)
            axes[i].xaxis.set_tick_params(labelsize=11)
            # axes[i].xaxis.set_title(labelsize='large')
            axes[i].set_xlim([0, 0.9])
            axes[i].yaxis.set_tick_params(labelsize=11)
            # axes[i].axvline(x=0.5)
            sns.barplot(
                ax=axes[i],
                y=level_2_tags_df["level_2"],
                x=level_2_tags_df["f_score"],
                palette=custom_palette,
            ).set(xlabel=None)
            plt.subplots_adjust(hspace=0.5)
            plt.xlabel("f1 score", fontsize=14)

        fig.suptitle(
            f"{one_level_0} results for each separate tag {context_type} {relabling_str}".replace(
                "_", " "
            ),
            fontsize=18,
        )

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(
            lines, labels, handles=handles, fontsize=12, loc=4, bbox_to_anchor=(0.9, 0)
        )

        plt.savefig(
            Path(args.output_data_dir)
            / f"results_visualization_{one_level_0}_{context_type}_{relabling_str}.png".replace(
                " ", "_"
            ),
            bbox_inches="tight",
        )


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
    test_set_results = _generate_results(
        transformer_model,
        test_data_one_tag_labeled.excerpt.tolist(),
        test_data_one_tag_labeled["target"].tolist(),
    )

    mlflow.log_metrics(_clean_results_for_logging(test_set_results, prefix="test"))
    mlflow.log_metrics(
        _clean_thresholds_for_logging(transformer_model.optimal_thresholds)
    )

    return transformer_model, test_set_results


def _log_results(
    final_results_dict: Dict,
    final_results_df: pd.DataFrame,
    context_type: str,
    relabling_str: str,
):
    # log results
    mlflow.log_metrics(
        _clean_results_for_logging(final_results_dict, prefix=context_type)
    )

    # save results dict
    with open(
        Path(args.output_data_dir)
        / f"test_set_results_{context_type}_{relabling_str}.json".replace(" ", "_"),
        "w",
    ) as fp:
        json.dump(final_results_dict, fp)

    # save results df
    final_results_df.to_csv(
        Path(args.output_data_dir)
        / f"test_set_results_{context_type}_{relabling_str}.csv".replace(" ", "_"),
        index=None,
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
    parser.add_argument("--apply_relabling", type=str, default="true")

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
            "_apply_relabling": args.apply_relabling,
        }

        model_args = {
            "BACKBONE_NAME": args.model_name,
            "TOKENIZER_NAME": args.tokenizer_name,
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
        # (
        #     trainable_data,
        #     projects_list_per_tag,
        #     out_of_context_test_data,
        # ) = _preprocess_df(all_data, args.min_entries_per_proj)
        (
            trainable_data,
            projects_list_per_tag,
        ) = _preprocess_df(all_data)

        ###############################     Apply Relabling     ###############################
        # if args.apply_relabling == "true":
        #     train_val_df = _relabel_sectors(
        #         train_val_df, projects_list_per_tag, model_args
        #     )

        #     train_val_df = _relabel_subsectors(train_val_df, model_args)

        ###############################  train backbone model   ###############################

        TRANSFORMER_MODEL_NAME = "all_tags_transformer_model"

        transformer_model, test_df = train_model(
            MODEL_NAME=TRANSFORMER_MODEL_NAME,
            trainable_data=trainable_data,
            hypertune_threshold_bool=True,
            f_beta=1,
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

        # log thresholds
        mlflow.log_metrics(
            _clean_thresholds_for_logging(transformer_model.optimal_thresholds)
        )

        _log_models()

        ##### test set results generation

        # test_datas = {"out of context": test_df, "out of context": out_of_context_test_data}
        context_type = "out of context"

        # if args.apply_relabling == "true":
        relabling_str = "with relabling"
        # else:
        #     relabling_str = "witout relabling"

        results_dict_one_context = _generate_test_set_results(
            transformer_model, test_df, projects_list_per_tag
        )

        results_df_one_context = _get_results_df_from_dict(
            results_dict_one_context, proportions
        )

        _log_results(
            results_dict_one_context,
            results_df_one_context,
            context_type,
            relabling_str,
        )

        _generate_visualization(results_df_one_context, context_type, relabling_str)
