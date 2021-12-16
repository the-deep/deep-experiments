import os
#setting tokenizers parallelism to false adds robustness when dploying the model
#os.environ["TOKENIZERS_PARALLELISM"] = "false" 
#dill import needs to be kept for more robustness in multimodel serialization
import dill
dill.extend(True)

import multiprocessing

import argparse
import logging

from pathlib import Path

import mlflow

from utils import (
    read_merge_data,
    preprocess_df,
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

    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--output_length", type=int, default=384)

    parser.add_argument("--model_name", type=str, default="microsoft/xtremedistil-l6-h384-uncased")
    parser.add_argument(
        "--tokenizer_name", type=str, default="microsoft/xtremedistil-l6-h384-uncased"
    )
    # parser.add_argument("--log_every_n_steps", type=int, default=10)
    # parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--training_names", type=str)
    parser.add_argument("--beta_f1", type=float, default=0.5)
    
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
    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run():
        train_params = {"batch_size": args.train_batch_size, "shuffle": True, "num_workers": 4}
        val_params = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": 4}

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
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "beta f1":args.beta_f1,
            "instance_type":args.instance_type,
            "n_gpu": gpu_nb
        }

        mlflow.log_params(params)

        pyfunc_prediction_wrapper = TransformersPredictionsWrapper()

        for column in training_columns:
            multiclass_bool = column != 'severity'
            keep_neg_examples = 'present' in column

            #sanity check on params
            mlflow.log_params({
                f'multiclass_bool_{column}': multiclass_bool,
                f'keep_neg_examples_bool_{column}': keep_neg_examples
            })

            train_df, val_df = preprocess_df(
                whole_df, 
                column, 
                multiclass_bool,
                keep_neg_examples)

            mlflow.log_params({
                f'nb_train_rows_{column}': train_df.shape[0],
                f'nb_val_rows_{column}': val_df.shape[0]
            })

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
                output_length=args.output_length,
                multiclass_bool=multiclass_bool,
                training_device=training_device,
                beta_f1 = args.beta_f1
            )
            
            model = model_trainer.train_model()
            pyfunc_prediction_wrapper.add_model(model, column)

            mlflow.log_params({f'lr_{column}': model.hparams.learning_rate})

        try:
            mlflow.pyfunc.log_model(
                python_model=pyfunc_prediction_wrapper,
                artifact_path="primary_tags_v1",
                conda_env=get_conda_env_specs(),  # python conda dependencies
                code_path=[
                    __file__,
                    "model.py",
                    "utils.py",
                    "inference.py", 
                    "generate_models.py",
                    "data.py",
                    "get_outputs_user.py"
                ],
                await_registration_for=600
            )
        except Exception as e:
            logging.info('pyfunc', e)
            
    raw_predictions = {}
    for tag_name, trained_model in pyfunc_prediction_wrapper.models.items():

        predictions_one_model = trained_model.custom_predict(test_df['excerpt'], testing=True)
        raw_predictions[tag_name] = predictions_one_model

    outputs = {
        'preds': raw_predictions,
        'thresholds': pyfunc_prediction_wrapper.thresholds
    }

    with open(Path(args.output_data_dir) / "logged_values.pickle", "wb") as f:
        dill.dump(outputs, f)