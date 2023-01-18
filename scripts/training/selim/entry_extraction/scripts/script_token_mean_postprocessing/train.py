import sys

sys.path.append(".")
import multiprocessing
import logging
from pathlib import Path
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
import argparse
import torch
import mlflow
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from ast import literal_eval
from model import TrainingExtractionModel, LoggedExtractionModel
from inference import EntryExtractionWrapper
from utils import (
    _clean_results_for_logging,
    hypertune_threshold,
    _generate_test_set_results,
    _clean_str_for_logging,
    _get_results_df_from_dict,
)
from prepare_data_for_training_job import DataPreparation
import time
import json

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


def _train_model():
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
        # enable_progress_bar=True,
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
        tag_token_proportions=tag_token_proportions,
        tag_cls_proportions=tag_cls_proportions,
        slice_length=args.max_len,
        extra_context_length=args.extra_context_length,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        tokens_focal_loss_gamma=args.tokens_focal_loss_gamma,
        cls_focal_loss_gamma=args.cls_focal_loss_gamma,
        proportions_pow=args.proportions_pow,
    )

    training_loader = training_model._get_loaders(
        train_dataset, train_params, training_mode=True
    )
    val_loader = training_model._get_loaders(
        val_dataset, val_params, training_mode=True
    )

    trainer.fit(training_model, training_loader, val_loader)

    # new model, used for logging, torch.nn.Module type
    # avoids logging errors
    training_model.eval()
    training_model.freeze()

    logged_extraction_model = LoggedExtractionModel(training_model)

    (
        val_results,
        optimal_thresholds_cls,
        optimal_thresholds_tokens,
    ) = hypertune_threshold(logged_extraction_model, val_loader, args.fbeta)
    logged_extraction_model.optimal_thresholds_cls = optimal_thresholds_cls
    logged_extraction_model.optimal_thresholds_tokens = optimal_thresholds_tokens
    logged_extraction_model.optimal_setups = {
        tagname: outputs["optimal_setup"] for tagname, outputs in val_results.items()
    }

    mlflow.log_metrics(_clean_results_for_logging(val_results))

    return logged_extraction_model


def _log_models():
    # gpu model logging
    # training is done on gpu, so no change to deploy in gpu
    gpu_prediction_wrapper = EntryExtractionWrapper(logged_extraction_model)

    mlflow.pyfunc.log_model(
        python_model=gpu_prediction_wrapper,
        artifact_path="entry-extraction-gpu",
        conda_env=get_conda_env_specs(),  # python conda dependencies
        code_path=[
            __file__,
            "model.py",
            "data.py",
            "inference.py",
            "utils.py",
            "merge_leads_excerpts.py",
            "requirements.txt",
        ],  # file dependencies
    )

    # cpu model logging
    logged_extraction_model.cpu()
    cpu_prediction_wrapper = EntryExtractionWrapper(logged_extraction_model)

    mlflow.pyfunc.log_model(
        python_model=cpu_prediction_wrapper,
        artifact_path="entry-extraction-cpu",
        conda_env=get_conda_env_specs(),  # python conda dependencies
        code_path=[
            __file__,
            "model.py",
            "data.py",
            "inference.py",
            "utils.py",
            "merge_leads_excerpts.py",
            "requirements.txt",
        ],  # file dependencies
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--tokenizer_name_or_path", type=str)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--extra_context_length", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--dataloader_num_workers", type=int, default=6)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--instance_type", type=str)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fbeta", type=float, default=1.4)
    parser.add_argument("--tokens_focal_loss_gamma", type=float, default=2)
    parser.add_argument("--cls_focal_loss_gamma", type=float, default=1)
    parser.add_argument("--proportions_pow", type=float, default=1)

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

    # load data
    full_data = pd.read_pickle(f"{args.data_dir}/data.pickle")

    data = literal_eval(full_data.iloc[0]["data"])
    tagname_to_tagid = literal_eval(full_data.iloc[0]["tagname_to_tagid"])

    preprocessed_data = DataPreparation(
        leads_dict=data,
        tagname_to_tagid=tagname_to_tagid,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
    ).final_outputs

    train_dataset = preprocessed_data["train"]
    test_in_context_dataset = preprocessed_data["test_in_context"]
    test_out_of_context_dataset = preprocessed_data["test_out_of_context"]
    val_dataset = preprocessed_data["val"]
    tag_token_proportions = preprocessed_data["tag_token_proportions"]
    tag_cls_proportions = preprocessed_data["tag_cls_proportions"]

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
        # log token proportions
        token_proportions = {
            tagname: round(tag_token_proportions[tag_id].item(), 2)
            for tagname, tag_id in tagname_to_tagid.items()
        }

        # log cls proportions
        cls_proportions = {
            tagname: round(tag_cls_proportions[tag_id].item(), 2)
            for tagname, tag_id in tagname_to_tagid.items()
        }
        tot_proportions = {"cls": cls_proportions, "tokens": token_proportions}

        n_leads_per_category = {
            "_n_leads_train": len(train_dataset),
            "_n_leads_val": len(val_dataset),
            "_n_leads_test_out_of_context": len(test_out_of_context_dataset),
            "_n_leads_test_in_context_dataset": len(test_in_context_dataset),
        }
        mlflow.log_params(n_leads_per_category)

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
            "name_model": args.model_name_or_path,
            "name_tokenizer": args.tokenizer_name_or_path,
            "instance_type": args.instance_type,
            "n_gpu": gpu_nb,
            "max_len": args.max_len,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "fbeta": args.fbeta,
            "focal_loss_gamma_tokens": args.tokens_focal_loss_gamma,
            "focal_loss_gamma_cls": args.cls_focal_loss_gamma,
            "proportions_pow": args.proportions_pow,
        }

        mlflow.log_params(model_params)

        # train model using pytorch lightning
        logged_extraction_model = _train_model()

        # This class is logged as a pickle artifact and used at inference time
        _log_models()

        # Generate test set results
        start_test_predictions = time.process_time()

        test_datasets = {
            "test_in_context": test_in_context_dataset,
            "test_out_of_context": test_out_of_context_dataset,
        }

        for context_name, test_dataset in test_datasets.items():

            results_test_set, n_total_test_sentences = _generate_test_set_results(
                logged_extraction_model, test_dataset, args.fbeta
            )

            end_test_predictions = time.process_time()
            test_set_results_generation_time = (
                end_test_predictions - start_test_predictions
            )
            mlflow.log_metrics(
                {
                    "_time_test_set_predictions_per_sentence": round(
                        test_set_results_generation_time / n_total_test_sentences, 4
                    )
                }
            )
            mlflow.log_metrics(
                _clean_results_for_logging(results_test_set, prefix=context_name)
            )
            with open(
                Path(args.output_data_dir) / f"{context_name}_results_predictions.json",
                "w",
            ) as f:
                json.dump(results_test_set, f)

            test_results_as_df = _get_results_df_from_dict(
                results_test_set, tot_proportions
            )
            test_results_as_df.to_csv(
                Path(args.output_data_dir) / f"{context_name}_results.csv", index=None
            )
