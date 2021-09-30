import os
import re
import argparse
import logging
import pickle

import dill

import pandas as pd
from pathlib import Path

import mlflow

from utils import (
    read_merge_data,
    preprocess_df,
    stats_train_test
)

from inference import TransformersPredictionsWrapper
from generate_models import CustomTrainer

##

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os





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
    parser.add_argument("--balance_trainig_data", type=bool, default=False)

    parser.add_argument("--model_name", type=str, default="microsoft/xtremedistil-l6-h384-uncased")
    parser.add_argument(
        "--tokenizer_name", type=str, default="microsoft/xtremedistil-l6-h384-uncased"
    )
    # parser.add_argument("--log_every_n_steps", type=int, default=10)
    # parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--method_language", type=str, default="keep all")
    parser.add_argument("--training_names", type=str, default="sectors,subpillars_2d,subpillars_1d")


    parser.add_argument("--train_with_all_positive_examples", type=bool, default=False)
    parser.add_argument("--multiclass_bool", type=bool, default=True)
    parser.add_argument("--proportion_negative_examples_train_df", type=float, default=0.01)
    #parser.add_argument("--log_models_bool", type=bool, default=True)
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

    
    all_dataframe = read_merge_data(
        args.training_dir, args.val_dir, data_format="pickle"
    )

    training_columns = args.training_names.split(',')
    

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
            "augmentation_performed": "translation",
            "weighted_loss": "sqrt weights",
            "train with whole dataset":args.train_with_all_positive_examples,
            "balance data":args.balance_trainig_data,
            "proportion negative examples train df":args.proportion_negative_examples_train_df
        }

        mlflow.log_params(params)

        tot_time = 0
        tot_nb_rows_predicted = 0
        all_results = {}
        prediction_wrapper = TransformersPredictionsWrapper()
        for column in training_columns:

            train_df, val_df = preprocess_df(
                all_dataframe, 
                column, 
                train_with_all_positive_examples=args.train_with_all_positive_examples, 
                proportion_negative_examples_train_df=args.proportion_negative_examples_train_df)

            model_trainer = CustomTrainer(
                train_dataset=train_df,
                val_dataset=val_df,
                MODEL_DIR=args.model_dir,
                MODEL_NAME=args.model_name,
                TOKENIZER_NAME=args.tokenizer_name,
                training_column=column,
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
            model = model_trainer.train_model()
            time_for_predictions, indexes, logit_predictions, y_true = model.hypertune_threshold()
            prediction_wrapper.add_model(model=model, model_name=column)
            optimal_metrics = model.custom_eval(logit_predictions, y_true)

            tot_time += time_for_predictions
            tot_nb_rows_predicted += y_true.shape[0]

            results_column = {
                'indexes': indexes,
                'logit_predictions': logit_predictions,
                'groundtruth': y_true,
                'thresholds': model.optimal_thresholds,
                'optimal_metrics': optimal_metrics
            }
            all_results[column] = results_column

            try:
                mlflow.log_metrics(optimal_metrics)
            except Exception:
                pass

            #logging metrcs to mlflow
            proportions = stats_train_test(train_df, val_df, column)
            mlflow.log_params(proportions)
            params.update(proportions)

            model_threshold_names = list(model.optimal_thresholds.keys())
            model_threshold_values = list(model.optimal_thresholds.values())

            for i in range(len(model_threshold_names)):
                try:
                    name = model_threshold_names[i]
                    cleaned_name = re.sub("[^0-9a-zA-Z]+", "_", name)
                    final_name = 'threshold_' + model.column_name + "_" + cleaned_name
                    
                    mlflow.log_metric(final_name, model_threshold_values[i])

                except Exception:
                    pass

            """test_model = TransformersPredictionsWrapper()
            test_model.add_model(model, column)
            mlflow.pyfunc.log_model(
                python_model=test_model,
                artifact_path=f"pyfunc_model_{column}",
                conda_env=get_conda_env_specs(),  # python conda dependencies
                code_path=[
                    __file__,
                    "model.py",
                    "utils.py",
                    "inference.py", 
                    "generate_models.py",
                    "data.py",
                ]  # file dependencies
            )"""
            

        nb_sentences_per_second = tot_nb_rows_predicted / tot_time
        mlflow.log_metric("nb sentences per second to predict all tags", nb_sentences_per_second)

        mlflow.pyfunc.log_model(
            python_model=prediction_wrapper,
            artifact_path="pyfunc_models_all",
            conda_env=get_conda_env_specs(),  # python conda dependencies
            code_path=[
                __file__,
                "model.py",
                "utils.py",
                "inference.py", 
                "generate_models.py",
                "data.py",
            ]  # file dependencies
        )



    outputs = {
        'parameters':params,
        'outputs':all_results 
    }
    with open(Path(args.output_data_dir) / "logged_values.pickle", "wb") as f:
        pickle.dump(outputs, f)
