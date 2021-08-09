from collections import OrderedDict
from typing import Dict

import argparse
from pathlib import Path

import pandas as pd

import torch
import mlflow


def revdict(d: Dict):
    """Creates a reverse dictionary (value:key) from a given dictionary (key:value)"""
    r = {}
    for k in d:
        r[d[k]] = k
    return r


def str2bool(v):
    """Maps given string into a boolean value"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("1", "y", "t", "yes", "true"):
        return True
    elif v.lower() in ("0", "n", "f", "no", "false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2list(v, sep=","):
    """Maps given string into a list of string with separator"""
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return v.split(sep)
    else:
        raise argparse.ArgumentTypeError("String value expected.")


def get_conda_env_specs():
    requirement_file = str(Path(__file__).parent / "requirements.txt")
    with open(requirement_file, "r") as f:
        requirements = f.readlines()
    requirements = [x.replace("\n", "") for x in requirements]

    default_env = mlflow.pytorch.get_default_conda_env()
    pip_dependencies = default_env["dependencies"][2]["pip"]
    pip_dependencies.extend(requirements)
    return default_env


def read_dataframe(path: str, **kwargs):
    """Reads a Pandas DataFrame respecting the file extension"""

    if path.endswith(".pickle"):
        return pd.read_pickle(path, **kwargs)
    if path.endswith(".csv"):
        return pd.read_csv(path, **kwargs)
    if path.endswith(".xlsx"):
        return pd.read_excel(path, **kwargs)
    raise "Unknown data format"


def build_mlp(
    depth: int,
    in_features: int,
    middle_features: int,
    out_features: int,
    bias: bool = True,
    batchnorm: bool = True,
    final_norm: bool = False,
) -> torch.nn.Sequential:
    """Builds a multi-layer perceptron network"""

    # initial dense layer
    layers = []
    layers.append(
        (
            "linear_1",
            torch.nn.Linear(in_features, out_features if depth == 1 else middle_features),
        )
    )

    #  iteratively construct batchnorm + relu + dense
    for i in range(depth - 1):
        layers.append((f"batchnorm_{i+1}", torch.nn.BatchNorm1d(num_features=middle_features)))
        layers.append((f"relu_{i+1}", torch.nn.ReLU()))
        layers.append(
            (
                f"linear_{i+2}",
                torch.nn.Linear(
                    middle_features,
                    out_features if i == depth - 2 else middle_features,
                    False if i == depth - 2 else bias,
                ),
            )
        )

    if final_norm:
        layers.append((f"batchnorm_{depth}", torch.nn.BatchNorm1d(num_features=out_features)))

    # return network
    return torch.nn.Sequential(OrderedDict(layers))
