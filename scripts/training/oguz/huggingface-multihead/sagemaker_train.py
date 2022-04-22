import os
import sys
import argparse

# import main folder for imports
sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd

import sagemaker
from sagemaker.pytorch import PyTorch

from deep.constants import DEV_BUCKET, MLFLOW_SERVER, SAGEMAKER_ROLE
from deep.utils import formatted_time

# get args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="1D",
    choices=["1D", "2D"],
)
parser.add_argument("--debug", action="store_true", default=False)
args, _ = parser.parse_known_args()

# create SageMaker session
sess = sagemaker.Session(default_bucket=DEV_BUCKET.name)
job_name = f"{args.task}-train-{formatted_time()}"

# load dataset
dataset_version = "0.7.1"
target_field = "subpillars_1d" if args.task == "1D" else "subpillars_2d"
train_df = pd.read_csv(f"data/frameworks_data/data_v{dataset_version}/train_v{dataset_version}.csv")
test_df = pd.read_csv(f"data/frameworks_data/data_v{dataset_version}/test_v{dataset_version}.csv")

# resample if debug
if args.debug:
    train_df = train_df.sample(n=1000)
    test_df = test_df.sample(n=1000)

# upload dataset to s3
input_path = DEV_BUCKET / "training" / "input_data" / job_name  # Do not change this
train_path = str(input_path / "train_df.pickle")
test_path = str(input_path / "test_df.pickle")

train_df.to_pickle(
    train_path, protocol=4
)  # protocol 4 is necessary, since SageMaker uses python 3.6
test_df.to_pickle(test_path, protocol=4)

# hyperparameters for the run
hyperparameters = {
    "epochs": 10,
    "model_name": "distilbert-base-uncased",
    "tracking_uri": MLFLOW_SERVER,
    "experiment_name": f"{args.task}-multihead-transformers",
    "loss": "ce",
    "iterative": False,
    "pooling": False,
    "save_model": False,
    "num_layers": 1,
    "split": target_field,
    "target": target_field,
    "weighting": "inverse",
    "learning_rate": 5e-5,
}

# create SageMaker estimator
estimator = PyTorch(
    entry_point="train.py",
    source_dir=str("scripts/training/oguz/huggingface-multihead"),
    output_path=str(DEV_BUCKET / "models/"),
    code_location=str(input_path),
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role=SAGEMAKER_ROLE,
    framework_version="1.8",
    py_version="py36",
    hyperparameters=hyperparameters,
    job_name=job_name,
)

# set arguments
fit_arguments = {"train": str(input_path), "test": str(input_path)}

# fit the estimator
estimator.fit(fit_arguments, job_name=job_name)
