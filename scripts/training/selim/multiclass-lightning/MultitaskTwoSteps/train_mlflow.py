import os

# setting tokenizers parallelism to false adds robustness when dploying the model
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dill import needs to be kept for more robustness in multimodel serialization
# import dill
# dill.extend(True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import multiprocessing
import argparse

from pathlib import Path
import mlflow
import copy
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np

from utils import (
    read_merge_data,
    preprocess_df,
    clean_name_for_logging,
    get_tagname_to_id,
    get_loss_alphas,
)
from TransformerModel import TrainingTransformer, LoggedTransformerModel
from MLPModel import TrainingMLP, LoggedMLPModel
from Inference import ClassificationInference

import logging

logging.basicConfig(level=logging.INFO)
# logging.getLogger().setLevel(logging.DEBUG)


def train_model(
    train_dataset,
    val_dataset,
    MODEL_DIR: str,
    MODEL_NAME: str,
    dropout_rate: float,
    train_params,
    val_params,
    gpu_nb: int,
    training_type: str,  # in ['transformer', 'MLP']
    BACKBONE_NAME: str = None,
    TOKENIZER_NAME: str = None,
    MAX_EPOCHS: int = 5,
    max_len: int = 128,
    n_freezed_layers: int = 0,
    weight_decay=0.02,
    output_length=384,
    learning_rate=3e-5,
    training_device: str = "cuda",
    f_beta: float = 0.8,
):
    """
    function to train one model
    """
    assert training_type in ["Transformer", "MLP"]

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=1 + MAX_EPOCHS // 3, mode="min"
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
        progress_bar_refresh_rate=40,
        profiler="simple",
        log_gpu_memory=True,
        weights_summary=None,
        gpus=gpu_nb,
        precision=16,
        accumulate_grad_batches=1,
        max_epochs=MAX_EPOCHS,
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

    if training_type == "Transformer":
        targets_list = train_dataset["target"].tolist()
    else:
        targets_list = train_dataset["y"]

    tagname_to_tagid = get_tagname_to_id(targets_list)
    loss_alphas = get_loss_alphas(tagname_to_tagid, targets_list)

    if training_type == "Transformer":
        model = TrainingTransformer(
            model_name_or_path=BACKBONE_NAME,
            tokenizer_name_or_path=TOKENIZER_NAME,
            tagname_to_tagid=tagname_to_tagid,
            loss_alphas=loss_alphas,
            val_params=val_params,
            gpus=gpu_nb,
            plugin="deepspeed_stage_3_offload",
            accumulate_grad_batches=1,
            max_epochs=MAX_EPOCHS,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            output_length=output_length,
            learning_rate=learning_rate,
            training_device=training_device,
            max_len=max_len,
            n_freezed_layers=n_freezed_layers,
        )

    else:
        model = TrainingMLP(
            val_params=val_params,
            tagname_to_tagid=tagname_to_tagid,
            loss_alphas=loss_alphas,
            gpus=gpu_nb,
            plugin="deepspeed_stage_3_offload",
            accumulate_grad_batches=1,
            max_epochs=MAX_EPOCHS,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            output_length=output_length,
            learning_rate=learning_rate,
            training_device=training_device,
        )

    train_dataloader = model.get_loaders(train_dataset, train_params)
    val_dataloader = model.get_loaders(val_dataset, val_params)

    trainer.fit(model, train_dataloader, val_dataloader)
    model.eval()
    model.freeze()

    # create new data structure, containing only needed information for deployment used for logging models
    if training_type == "Transformer":
        del model.model.output_layer
        returned_model = LoggedTransformerModel(model)

    else:
        returned_model = LoggedMLPModel(model)
        # hypertune decision boundary thresholds for each label
        returned_model.optimal_scores = returned_model.hypertune_threshold(
            val_dataset, f_beta
        )

    # move model to cpu for deployment
    return returned_model.to(torch.device("cpu"))


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
            {str(af_id): np.round(af_prop, 2) for af_id, af_prop in afs_counts.items()}
        )

        train_df, val_df = preprocess_df(train_val_df, args.relabeled_columns)

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

        # train new model, AF wise
        min_af_count = train_df.shape[0] // 20
        trained_afs = [
            af_id
            for af_id, one_af_count in afs_counts.items()
            if one_af_count > min_af_count
        ]
        trained_afs.append("all")

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
                MAX_EPOCHS=args.epochs,  # * 5,
                dropout_rate=args.dropout,
                weight_decay=args.weight_decay,
                learning_rate=args.learning_rate * 100,
                output_length=args.output_length,
                training_device=training_device,
                f_beta=args.f_beta,
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
                    f"a_mean_overall_{metric}_{one_AF}", np.mean(list(scores.values()))
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
                        mlflow.log_metric(
                            f"aa_mean_{task_name}_{metric}_{one_AF}", np.mean(vals)
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
            ],
            await_registration_for=600,
        )

    # TESTING
    # save time taken

    results_df = pd.DataFrame()
    afs_list_test_set = test_df.analysis_framework_id.unique()
    for one_AF in afs_list_test_set:
        if one_AF in trained_afs:
            model_af = one_AF
        else:
            model_af = "all"

        test_df_one_af = test_df[test_df.analysis_framework_id == one_AF]
        backbone_outputs_one_af = transformer_model.get_transformer_outputs(
            test_df_one_af["excerpt"]
        )
        predictions_test_set_one_af = MLP_prediction_wrapper.models[
            model_af
        ].custom_predict({"X": backbone_outputs_one_af}, testing=True)
        test_df_one_af["predictions"] = predictions_test_set_one_af
        results_df = results_df.append(test_df_one_af[["entry_id", "predictions"]])

    # save thresholds
    thresholds = {
        model_af: logged_models[model_af].optimal_thresholds for model_af in trained_afs
    }

    with open(Path(args.output_data_dir) / "logged_values.pickle", "wb") as f:
        dill.dump(thresholds, f)

    # save predictions df
    results_df.to_csv(Path(args.output_data_dir) / "results_df.csv", index=None)
