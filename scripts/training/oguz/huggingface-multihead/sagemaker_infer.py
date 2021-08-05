import os
import sys
import argparse

# import main folder for imports
sys.path.append(os.path.abspath(os.getcwd()))

import pandas as pd

import sagemaker
from sagemaker.pytorch import PyTorch

from deep.constants import DEV_BUCKET, SAGEMAKER_ROLE
from deep.utils import formatted_time

# get args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="1D",
    choices=["1D", "2D"],
)
parser.add_argument("--dataset", type=str, default=None, required=True)
parser.add_argument("--model_uri", type=str, default=None, required=True)
parser.add_argument("--source", type=str, default="excerpt")
parser.add_argument("--debug", action="store_true", default=False)
args, _ = parser.parse_known_args()

# create SageMaker session
sess = sagemaker.Session(default_bucket=DEV_BUCKET.name)

# job and experiment names
job_name = f"{args.task}-infer-{formatted_time()}"

# load dataset
infer_df = pd.read_csv(args.dataset)
infer_df.rename(columns={args.source: "excerpt"}, inplace=True)
if args.debug:
    infer_df = infer_df.sample(n=1000)

# upload dataset to s3
input_path = DEV_BUCKET / "inference" / "input_data" / job_name  # Do not change this
infer_path = str(input_path / "infer_df.pickle")

infer_df.to_pickle(
    infer_path, protocol=4
)  # protocol 4 is necessary, since SageMaker uses python 3.6

# hyperparameters for inference
hyperparameters = {
    "model_uri": args.model_uri,
}

# create SageMaker estimator
estimator = PyTorch(
    entry_point="infer.py",
    source_dir=str("scripts/training/oguz/huggingface-multihead"),
    output_path=str(DEV_BUCKET / "predictions/"),
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

# transform the estimator
estimator.fit(fit_arguments, job_name=job_name)
