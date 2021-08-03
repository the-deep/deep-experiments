from collections import OrderedDict
from typing import Dict

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
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


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
):
    """
    Original implementation from
        https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
