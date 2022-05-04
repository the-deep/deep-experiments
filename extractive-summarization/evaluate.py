import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
from transformers.hf_argparser import HfArgumentParser
from functools import partial
from types import SimpleNamespace
import pickle

from train import encode, Metrics
from model import BasicModel


@dataclass
class Args:
    model_name_or_path: str
    data_path: str
    max_full_length: int = 4096
    extra_context_length: int = 0
    max_length: int = 256
    sentence_edit_threshold: int = math.inf


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h384-uncased")
    model = BasicModel(
        "microsoft/xtremedistil-l6-h384-uncased",
        num_labels=5,
        tokenizer=tokenizer,
        token_loss_weight=None,
        loss_weights=None,
        extra_context_length=args.extra_context_length,
        slice_length=args.max_length,
    )
    model.load_state_dict(torch.load(args.model_name_or_path))
    excerpts_dict = pickle.load(open("excerpts_dict.pkl", "rb"))

    data = load_dataset("json", data_files=args.data_path, split="train")
    data = data.filter(lambda x: len(x["excerpts"]) > 0)
    (eval_indices,) = np.where(~np.array(data["train"]))
    data = data.select(eval_indices)

    data = data.map(
        partial(encode, tokenizer=tokenizer, args=args, excerpts_dict=excerpts_dict),
    )
    data = data.add_column("labels", data[f"token_labels"])
    data = data.add_column("labels_mask", data[f"token_labels_mask"])

    batch_size = 32
    device = "cuda"

    model.to(device)
    preds = torch.zeros((len(data), args.max_full_length, 5))
    labels = torch.zeros_like(preds)

    with torch.inference_mode():
        for i in range(math.ceil(len(data) / batch_size)):
            start, end = i * batch_size, min((i + 1) * batch_size, len(data))

            preds[start:end] = model(
                **{
                    k: torch.tensor(v, device=device)
                    for k, v in data[start:end].items()
                    if k
                    in [
                        "input_ids",
                        "attention_mask",
                        "token_labels",
                        "token_labels_mask",
                        "sentence_labels",
                        "sentence_labels_mask",
                    ]
                }
            )["logits"]
            labels[start:end] = torch.tensor(data[start:end]["token_labels"]) * 100.0

    metrics = Metrics(
        data,
        tokenizer=tokenizer,
    )
    metrics(
        SimpleNamespace(predictions=preds.cpu().numpy()),
        label_names_to_evaluate=["is_relevant"],
        max_n_visualize=None,
        visualize_out="predictions.html",
    )
    metrics(
        SimpleNamespace(predictions=labels.cpu().numpy()),
        label_names_to_evaluate=["is_relevant"],
        max_n_visualize=None,
        visualize_out="labels.html",
    )
