import os
import sys

# import main folder for imports
sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd

import sagemaker
from sagemaker.pytorch import PyTorch

from deep.constants import DEV_BUCKET, MLFLOW_SERVER, SAGEMAKER_ROLE
from deep.utils import formatted_time

# create SageMaker session
sess = sagemaker.Session(default_bucket=DEV_BUCKET.name)
job_name = f"1D-test-{formatted_time()}"

# load dataset
train_df = pd.read_csv("data/frameworks_data/data_v0.5/data_v0.5_train.csv")
val_df = pd.read_csv("data/frameworks_data/data_v0.5/data_v0.5_val.csv")

if "DEBUG" in os.environ and os.environ["DEBUG"]:
    train_df = train_df.sample(n=1000)
    val_df = val_df.sample(n=1000)

# upload dataset to s3
input_path = DEV_BUCKET / "training" / "input_data" / job_name  # Do not change this
train_path = str(input_path / "train_df.pickle")
val_path = str(input_path / "test_df.pickle")

train_df.to_pickle(
    train_path, protocol=4
)  # protocol 4 is necessary, since SageMaker uses python 3.6
val_df.to_pickle(val_path, protocol=4)

# hyperparameters for the run
hyperparameters = {
    "epochs": 10,
    "model_name": "distilbert-base-uncased",
    "tracking_uri": MLFLOW_SERVER,
    "experiment_name": "1D-multihead-transformers",
    "loss": "ce",
    "iterative": False,
    "pooling": False,
    "save_model": False,
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
