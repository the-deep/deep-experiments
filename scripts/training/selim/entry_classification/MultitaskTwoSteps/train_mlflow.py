import os

# setting tokenizers parallelism to false adds robustness when dploying the model
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dill import needs to be kept for more robustness in multimodel serialization
# import dill
# dill.extend(True)

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

from utils import (
    read_merge_data,
    preprocess_df,
    clean_name_for_logging,
    custom_eval,
    preprocess_text,
    get_n_tokens,
)

from ModelTraining import train_model
from ModelsExplainability import MultiLabelClassificationExplainer
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
    parser.add_argument("--relabeled_columns", type=str, default="none")
    parser.add_argument("--apply_preprocessing", type=str, default="false")
    parser.add_argument("--predictions_on_test_set", type=str, default="true")
    parser.add_argument("--explainability", type=str, default="true")
    parser.add_argument("--delete_long_excerpts", type=str, default="false")
    parser.add_argument("--loss_gamma", type=float, default=2)
    parser.add_argument("--proportions_pow", type=float, default=1)

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

    train_val_df, test_df = read_merge_data(
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
            "relabeled_columns": args.relabeled_columns,
        }

        mlflow.log_params(params)
        mlflow.log_param("train_batch_size", args.train_batch_size)

        train_df, val_df = preprocess_df(train_val_df, args.relabeled_columns)

        # preprocess text
        if args.apply_preprocessing == "true":
            train_df["excerpt"] = train_df["excerpt"].apply(preprocess_text)
            val_df["excerpt"] = val_df["excerpt"].apply(preprocess_text)

        if args.delete_long_excerpts == "true":

            n_tokens_train = get_n_tokens(train_df.excerpt.tolist())
            n_tokens_val = get_n_tokens(val_df.excerpt.tolist())

            train_df = train_df.iloc[n_tokens_train <= args.max_len]
            val_df = val_df.iloc[n_tokens_val <= args.max_len]

        # initialize models dict
        logged_models = {}

        ###############################  train backbone model   ###############################
        TRANSFORMER_MODEL_NAME = "transformer_model"

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
            training_type="Transformer",
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
        )

        Transformer_model_path = str(Path(args.model_dir) / TRANSFORMER_MODEL_NAME)
        mlflow.log_param("transfmoer_model_path", Transformer_model_path)

        # a deepcopy of the returned model is needed in order not to have pickling issues during logging
        logged_models["backbone"] = copy.deepcopy(transformer_model)

        # get predictions of backbone (train, val)
        train_outputs_backbone = transformer_model.get_transformer_outputs(
            train_df.excerpt
        )
        val_outputs_backbone = transformer_model.get_transformer_outputs(val_df.excerpt)

        train_params_MLP = {
            "batch_size": args.train_batch_size // 2,
            "shuffle": True,
            "num_workers": 4,
            "drop_last": True,
        }
        val_params_MLP = {
            "batch_size": args.val_batch_size * 2,
            "shuffle": False,
            "num_workers": 4,
            "drop_last": False,
        }

        trained_afs = ["all"]

        af_in_columns_bool = "analysis_framework_id" in train_val_df.columns

        if af_in_columns_bool:
            # get analysis framework counts
            afs_counts = (
                pd.concat(
                    [
                        train_val_df[["analysis_framework_id"]],
                        test_df[["analysis_framework_id"]],
                    ]
                )["analysis_framework_id"]
                .value_counts()
                .to_dict()
            )
            mlflow.log_params(
                {
                    str(af_id): np.round(af_prop, 2)
                    for af_id, af_prop in afs_counts.items()
                }
            )
            min_af_count = train_df.shape[0] // 20
            trained_afs += [
                int(af_id)
                for af_id, one_af_count in afs_counts.items()
                if one_af_count > min_af_count
            ]

        # train new model, AF wise
        for one_AF in trained_afs:

            # each AF alone
            if type(one_AF) is not str:
                mask_train_af = (train_df.analysis_framework_id == one_AF).tolist()
                mask_val_af = (val_df.analysis_framework_id == one_AF).tolist()

                train_df_one_af = {
                    "X": train_outputs_backbone[mask_train_af],
                    "y": train_df[mask_train_af].target.tolist(),
                }
                val_df_one_af = {
                    "X": val_outputs_backbone[mask_val_af],
                    "y": val_df[mask_val_af].target.tolist(),
                }
            # all AFs
            else:
                train_df_one_af = {
                    "X": train_outputs_backbone,
                    "y": train_df.target.tolist(),
                }
                val_df_one_af = {
                    "X": val_outputs_backbone,
                    "y": val_df.target.tolist(),
                }
            assert len(train_df_one_af["X"]) == len(train_df_one_af["y"])
            assert len(val_df_one_af["X"]) == len(val_df_one_af["y"])

            # train model
            MLP_MODEL_NAME = f"MLP_model_{one_AF}"

            MLP_model_one_af = train_model(
                MODEL_DIR=args.model_dir,
                MODEL_NAME=MLP_MODEL_NAME,
                train_dataset=train_df_one_af,
                val_dataset=val_df_one_af,
                train_params=train_params_MLP,
                val_params=val_params_MLP,
                training_type="MLP",
                gpu_nb=gpu_nb,
                MAX_EPOCHS=args.epochs ** 2,  # 1 for trainig purposes
                dropout_rate=args.dropout,
                weight_decay=args.weight_decay,
                learning_rate=args.learning_rate * 100,
                output_length=args.output_length,
                training_device=training_device,
                f_beta=args.f_beta,
                loss_gamma=args.loss_gamma,
                proportions_pow=args.proportions_pow,
            )

            mlflow.log_metrics(
                clean_name_for_logging(
                    MLP_model_one_af.optimal_thresholds,
                    context="thresholds",
                    af_id=one_AF,
                )
            )

            MLP_model_path = str(Path(args.model_dir) / MLP_MODEL_NAME)
            mlflow.log_param(f"MLP_model_path_{one_AF}", MLP_model_path)

            logged_models[one_AF] = copy.deepcopy(MLP_model_one_af)

            for metric, scores in MLP_model_one_af.optimal_scores.items():
                mlflow.log_metrics(clean_name_for_logging(scores, context=metric))
                mlflow.log_metric(
                    f"_mean_overall_{metric}_{one_AF}", np.mean(list(scores.values()))
                )
                if args.relabeled_columns == "none":
                    for task_name in [
                        "first_level_tags",
                        "secondary_tags",
                        "subpillars",
                    ]:
                        vals = [
                            one_score
                            for one_name, one_score in scores.items()
                            if task_name in one_name
                        ]
                        if len(vals) > 0:
                            mlflow.log_metric(
                                f"__mean_{task_name}_{metric}_{one_AF}", np.mean(vals)
                            )

        import dill

        dill.extend(True)

        MLP_prediction_wrapper = ClassificationInference(logged_models)

        mlflow.pyfunc.log_model(
            python_model=MLP_prediction_wrapper,
            artifact_path=f"two_steps_models",
            conda_env=get_conda_env_specs(),  # python conda dependencies
            code_path=[
                __file__,
                "loss.py",
                "utils.py",
                "Inference.py",
                "MLPModel.py",
                "TransformerModel.py",
                "pooling.py",
                "ModelsExplainability.py",
            ],
            await_registration_for=600,
        )

        ######################################## Testing #################################

        if args.predictions_on_test_set == "true":
            n_test_entries = len(test_df)

            results_df = pd.DataFrame()
            if af_in_columns_bool:
                afs_list_test_set = test_df.analysis_framework_id.apply(int).unique()

            total_tested_entries = 0
            start_predictions = time.process_time()
            for one_AF in trained_afs:

                if type(one_AF) is not str and one_AF in afs_list_test_set:
                    model_af = int(one_AF)
                    test_df_one_af = test_df[test_df.analysis_framework_id == one_AF]
                else:
                    model_af = "all"
                    if "analysis_framework_id" in test_df.columns:
                        test_df_one_af = test_df[
                            ~test_df.analysis_framework_id.isin(one_AF)
                        ]
                    else:
                        test_df_one_af = test_df

                total_tested_entries += test_df_one_af.shape[0]

                backbone_outputs_one_af = transformer_model.get_transformer_outputs(
                    test_df_one_af["excerpt"]
                )

                predictions_test_set_one_af = MLP_prediction_wrapper.models[
                    model_af
                ].custom_predict({"X": backbone_outputs_one_af}, testing=True)

                test_df_one_af[
                    "ratio_pred_threshold_all_labels"
                ] = predictions_test_set_one_af

                results_df = results_df.append(
                    test_df_one_af[
                        [
                            "entry_id",
                            "excerpt",
                            "ratio_pred_threshold_all_labels",
                            "target",
                        ]
                    ]
                )

            # save predictions df
            results_df.to_csv(Path(args.output_data_dir) / "results_df.csv", index=None)

            assert total_tested_entries == n_test_entries

            end_predictions = time.process_time()
            time_for_predictions = np.round(
                (end_predictions - start_predictions) / n_test_entries, 4
            )

            mlflow.log_metric("z_predictions_time_per_sentence", time_for_predictions)

        ################################### Explainability #######################################

        if args.explainability == "true":

            start_explainer = time.process_time()

            not_relevant_labels_list = [
                label
                for label, score in MLP_model_one_af.optimal_scores[
                    "f_beta_scores"
                ].items()
                if score < 0.6
            ]  # get_relevant_labels(train_val_df.target, 0)

            if test_df.shape[0] < 200:
                explainability_df = test_df
            else:
                explainability_df = pd.concat([train_val_df, test_df])[
                    ["entry_id", "lang", "excerpt", "target"]
                ].copy()
            explainability_df = explainability_df[explainability_df.lang == "en"]
            n_explainability_entries = explainability_df.shape[0]

            n_unperfect_predictions = 0
            total_explained_labels = 0

            cls_explainer = MultiLabelClassificationExplainer(
                logged_models["backbone"]  # .to(torch.device("cpu")),
            )

            interpretability_results = defaultdict(list)

            for i in range(n_explainability_entries):
                # each 100 sentences, log to mlflow the the sentence number and save the predictions
                if i % 100 == 0:
                    mlflow.log_metric("zz_iter_number", i)
                    with open(
                        Path(args.output_data_dir) / "explainability_results.pickle",
                        "wb",
                    ) as f:
                        dill.dump(interpretability_results, f)

                row_i = explainability_df.iloc[i]
                one_sentence = row_i["excerpt"]
                one_entry_id = row_i["entry_id"]
                groundtruth_one_row = custom_eval(row_i["target"])
                groundtruth_one_row = [
                    item
                    for item in groundtruth_one_row
                    if "first_level_tags->pillars" not in item
                ]

                if len(groundtruth_one_row) > 0:
                    attributions_one_entry = cls_explainer(
                        one_sentence, groundtruth_one_row
                    )
                    total_explained_labels += len(groundtruth_one_row)
                    for label_name, sentence in attributions_one_entry.items():
                        interpretability_results[label_name].append(
                            {"entry_id": one_entry_id, "sentence": sentence}
                        )

            end_explainer = time.process_time()
            # save time taken
            time_for_interpretability_per_sentence = np.round(
                (end_explainer - start_explainer) / n_test_entries, 2
            )

            mlflow.log_metric(
                "z_explainability_time_per_sentence",
                time_for_interpretability_per_sentence,
            )

            time_for_interpretability_per_label = np.round(
                (end_explainer - start_explainer) / total_explained_labels, 2
            )

            mlflow.log_metric(
                "z_explainability_time_per_label", time_for_interpretability_per_label
            )

    # save thresholds
    thresholds = {
        model_af: logged_models[model_af].optimal_thresholds for model_af in trained_afs
    }

    with open(Path(args.output_data_dir) / "logged_values.pickle", "wb") as f:
        dill.dump(thresholds, f)