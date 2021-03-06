import os
#setting tokenizers parallelism to false adds robustness when dploying the model
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

import re
import argparse
import logging
import timeit

#Importing pickle will cause an error in the model logging to mlflow
#import pickle
from ast import literal_eval
import multiprocessing

#dill import needs to be kept for more robustness in multimodel serialization
import dill
dill.extend(True)

from pathlib import Path

import mlflow

from utils import (
    read_merge_data,
    preprocess_df,
    stats_train_test
)
import torch

from inference import TransformersPredictionsWrapper
from generate_models import CustomTrainer


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
    parser.add_argument("--warmup_steps", type=int, default=10)
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
    parser.add_argument("--training_names", type=str, default="sectors,subpillars_2d,subpillars_1d")

    parser.add_argument("--training_mode", type=str, default='multiclass')
    parser.add_argument("--model_mode", type=str, default='train')
    parser.add_argument("--beta_f1", type=float, default=0.5)
    parser.add_argument("--numbers_augmentation", type=str, default="without")

    parser.add_argument("--proportions_negative_examples_test", type=str)
    parser.add_argument("--proportions_negative_examples_train", type=str)
    
    parser.add_argument("--instance_type", type=str, default="-")

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

    deployment_mode = args.model_mode=='deploy'
    augment_numbers = args.numbers_augmentation=='with'
    
    proportions_negative_examples_test = literal_eval(args.proportions_negative_examples_test)
    proportions_negative_examples_train = literal_eval(args.proportions_negative_examples_train)
    
    whole_df, test_df = read_merge_data(
        args.training_dir, args.val_dir, data_format="pickle"
    )

    training_columns = args.training_names.split(',')
    
    # Set remote mlflow server
    mlflow.set_tracking_uri(args.tracking_uri)
    # Set experiment name
    mlflow.set_experiment(args.experiment_name)
    # pytorch autolog automatically logs relevant data. DO NOT log the model, since
    # for NLP tasks we need a custom inference logic
    mlflow.pytorch.autolog(log_models=True)

    with mlflow.start_run():
        train_params = {"batch_size": args.train_batch_size, "shuffle": True, "num_workers": 1}

        val_params = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": 1}

        if torch.cuda.is_available():
            gpu_nb=1
            training_device='cuda'
        else:
            gpu_nb=0
            training_device='cpu'

        params = {
            "epochs": args.epochs,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout_rate,
            "threshold": args.pred_threshold,
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "purpose of run":args.model_mode,
            "beta f1":args.beta_f1,
            "numbers augmentation":args.numbers_augmentation,
            "instance_type":args.instance_type
        }

        mlflow.log_params(params)

        tot_time = 0

        tot_nb_rows_predicted = 0
        all_results = {}
        all_thresholds = {}
        pyfunc_prediction_wrapper = TransformersPredictionsWrapper()
        #pytorch_prediction_wrapper = PythonPredictor()
        for column in training_columns:

            multiclass_bool = column != 'severity'

            if multiclass_bool:
                prop_negative_examples_train_column = proportions_negative_examples_train[column]
                prop_negative_examples_test_column = proportions_negative_examples_test[column]
            else:
                prop_negative_examples_train_column = 0
                prop_negative_examples_test_column = 0

            train_df, val_df = preprocess_df(
                whole_df, 
                column, 
                proportion_negative_examples_train_df=prop_negative_examples_train_column,
                proportion_negative_examples_val_df=prop_negative_examples_test_column,
                augment_numbers=augment_numbers)

            #logging metrcs to mlflow
            proportions = stats_train_test(train_df, val_df, column)
            mlflow.log_params(proportions)
            params.update(proportions)

            model_trainer = CustomTrainer(
                train_dataset=train_df,
                val_dataset=val_df,
                MODEL_DIR=args.model_dir,
                MODEL_NAME=args.model_name,
                TOKENIZER_NAME=args.tokenizer_name,
                training_column=column,
                gpu_nb=gpu_nb,
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
                multiclass_bool=multiclass_bool,
                training_device=training_device,
                beta_f1 = args.beta_f1
            )
            
            model = model_trainer.train_model()
            
            #pytorch_prediction_wrapper.add_model(model, column)
            pyfunc_prediction_wrapper.add_model(model, column)

            #there are thresholds only when we are working with multilabels
            #In single labels, we just take the max probability
            if multiclass_bool:
                model_threshold_names = list(model.optimal_thresholds.keys())
                model_threshold_values = list(model.optimal_thresholds.values())
                all_thresholds[column] = model.optimal_thresholds

                pyfunc_prediction_wrapper.add_threshold(model.optimal_thresholds, column)

                for i in range(len(model_threshold_names)):
                    try:
                        name = model_threshold_names[i]
                        cleaned_name = re.sub("[^0-9a-zA-Z]+", "_", name)
                        final_name = f"threshold_{column}_{cleaned_name}"
                        
                        mlflow.log_metric(final_name, model_threshold_values[i])

                    except Exception:
                        pass

            start = timeit.default_timer()
            results_one_col = model.custom_predict(
                test_df['excerpt'], testing=True
                )
            end = timeit.default_timer()
            all_results[column] = results_one_col

            tot_time += (end - start)

            """all_results[column] = results_test_set
            all_results_raw[column] = results_raw
            try:
                mlflow.log_metrics(results_test_set)
            except Exception:
                pass"""

        mlflow.log_metric('nb row per second', len(test_df) / tot_time)

        try:
            mlflow.pyfunc.log_model(
                python_model=pyfunc_prediction_wrapper,
                artifact_path="pyfunc_models_all",
                conda_env=get_conda_env_specs(),  # python conda dependencies
                code_path=[
                    __file__,
                    "model.py",
                    "utils.py",
                    "inference.py", 
                    "generate_models.py",
                    "data.py",
                    #"get_outputs_user.py"
                ]
            )
        except Exception as e:
            logging.info("PYFUNC", e)

        """try:
            mlflow.pytorch.log_model(
                pytorch_model=pytorch_prediction_wrapper,
                artifact_path="pytorch_model_all",
                conda_env=get_conda_env_specs(),  # python conda dependencies
                code_paths=[
                    __file__,
                    "model.py",
                    "utils.py",
                    "inference.py", 
                    "generate_models.py",
                    "data.py",
                    ],
                pickle_module=dill
        )
        except Exception as e:
            logging.info("PYTORCH", e)"""
        

    outputs = {
        'parameters':params,
        'results':all_results,
        'thresholds':all_thresholds,
        'val_set_ids':val_df.entry_id.unique().tolist(),
        'test_set_ids':test_df.entry_id.tolist()
    }
    with open(Path(args.output_data_dir) / "logged_values.pickle", "wb") as f:
        dill.dump(outputs, f)
