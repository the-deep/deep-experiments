import sys

sys.path.append(".")

import math
import logging
from typing import List
from pathlib import Path
from dataclasses import dataclass, field

import mlflow
import pytorch_lightning as pl
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from transformers.hf_argparser import HfArgumentParser
from torch.utils.data import DataLoader

from model import ExtractionModel
from data import (
    get_test_train_data,
    LABEL_NAMES,
    ExtractionDataset,
)
from inference import EntryExtractionWrapper

logging.basicConfig(level=logging.INFO)


@dataclass
class Args:
    model_name_or_path: str
    data_path: str
    excerpts_csv_path: str
    max_full_length: int
    max_length: int
    extra_context_length: int
    tracking_uri: str
    experiment_name: str
    n_separate_layers: int = 0
    separate_layer_groups: List[List[str]] = []
    token_loss_weight: float = 1.0
    sentence_edit_threshold: int = math.inf
    n_subsample: int = None
    compute_relevant_with_or: bool = False
    loss_weights: List[float] = field(
        default_factory=lambda: [1.0] + [0.0] * (len(LABEL_NAMES) - 1)
    )


def get_conda_env_specs():
    requirement_file = str(Path(__file__).parent / "requirements.txt")
    with open(requirement_file, "r") as f:
        requirements = f.readlines()
    requirements = [x.replace("\n", "") for x in requirements]

    default_env = mlflow.pytorch.get_default_conda_env()
    pip_dependencies = default_env["dependencies"][2]["pip"]
    pip_dependencies.extend(requirements)
    return default_env


def get_args():
    parser = HfArgumentParser([Args, TrainingArguments])

    (args, training_args) = parser.parse_json_file(sys.argv[1])
    return args, training_args


def get_separate_layer_groups(args):
    if args.separate_layer_groups is not None:
        separate_layer_groups = []

        for group in args.separate_layer_groups:
            separate_layer_groups.append([LABEL_NAMES.index(label_name) for label_name in group])
    else:
        separate_layer_groups = args.separate_layer_groups
    return separate_layer_groups


if __name__ == "__main__":
    logging.info("building training and testing datasets")

    args, training_args = get_args()
    train_data, test_data, val_data = get_test_train_data(args, training_args)
    train_dataset, test_dataset, val_dataset = (
        ExtractionDataset(train_data),
        ExtractionDataset(test_data),
        ExtractionDataset(val_data),
    )

    # Set remote mlflow server
    mlflow.set_tracking_uri(args.tracking_uri)
    # Set experiment name
    mlflow.set_experiment(args.experiment_name)
    # pytorch autolog automatically logs relevant data. DO NOT log the model, since
    # for NLP tasks we need a custom inference logic
    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run():

        train_params = {"batch_size": args.train_batch_size, "shuffle": True, "num_workers": 0}
        val_params = {"batch_size": args.eval_batch_size, "shuffle": False, "num_workers": 0}

        training_loader = DataLoader(train_dataset, **train_params)
        val_loader = DataLoader(val_dataset, **val_params)

        params = {
            "train": train_params,
            "val": val_params,
        }
        mlflow.log_params(params)  # Logging example

        logging.info("training model")

        trainer = pl.Trainer(gpus=1, max_epochs=args.epochs)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = ExtractionModel(
            args.model_name_or_path,
            tokenizer,
            num_labels=len(LABEL_NAMES),
            token_loss_weight=args.token_loss_weight,
            loss_weights=args.loss_weights,
            slice_length=args.max_length,
            extra_context_length=args.extra_context_length,
            n_separate_layers=args.n_separate_layers,
            separate_layer_groups=get_separate_layer_groups(args),
        )
        trainer.fit(model, training_loader, val_loader)

        # This class is logged as a pickle artifact and used at inference time
        prediction_wrapper = EntryExtractionWrapper(tokenizer, model)
        mlflow.pyfunc.log_model(
            python_model=prediction_wrapper,
            artifact_path="model",
            conda_env=get_conda_env_specs(),  # python conda dependencies
            code_path=[__file__, "model.py", "data.py", "inference.py"],  # file dependencies
        )
