import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass
import math
import numpy as np
from transformers.hf_argparser import HfArgumentParser
from functools import partial
from types import SimpleNamespace

from train import encode, Metrics, LABEL_NAMES
from train import Args as TrainArgs
from model import Model


@dataclass
class Args:
    model_path: str
    model_config_path: str
    data_path: str
    batch_size: int = 32
    n_subsample: int = 100
    output_format: str = "html"  # or 'json"
    use_categories: bool = False


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    (train_args,) = HfArgumentParser([TrainArgs]).parse_json_file(
        args.model_config_path
    )

    tokenizer = AutoTokenizer.from_pretrained(train_args.model_name_or_path)

    separate_layer_groups = []

    for group in train_args.separate_layer_groups:
        separate_layer_groups.append(
            [LABEL_NAMES.index(label_name) for label_name in group]
        )

    model = Model(
        train_args.model_name_or_path,
        num_labels=len(LABEL_NAMES),
        tokenizer=tokenizer,
        token_loss_weight=None,
        loss_weights=None,
        extra_context_length=train_args.extra_context_length,
        slice_length=train_args.max_length,
        n_separate_layers=train_args.n_separate_layers,
        separate_layer_groups=separate_layer_groups,
    )
    model.load_state_dict(torch.load(args.model_path))

    data = load_dataset("json", data_files=args.data_path, split="train")
    data = data.filter(lambda x: len(x["excerpts"]) > 0)
    if args.n_subsample is not None:
        eval_indices = np.arange(args.n_subsample)
        data = data.select(eval_indices)

    data = data.map(
        partial(encode, tokenizer=tokenizer, args=train_args, excerpts_dict=None),
    )
    data = data.add_column("labels", data[f"token_labels"])
    data = data.add_column("labels_mask", data[f"token_labels_mask"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    preds = torch.zeros((len(data), train_args.max_full_length, len(LABEL_NAMES)))
    labels = torch.zeros_like(preds)

    with torch.inference_mode():
        for i in range(math.ceil(len(data) / args.batch_size)):
            start, end = i * args.batch_size, min((i + 1) * args.batch_size, len(data))

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
            # the metrics take logits as input, so we have to convert the 0/1 labels to logits
            # i.e. 0 -> -inf, 1 -> inf
            labels[start:end] = (
                torch.tensor(data[start:end]["token_labels"]) - 0.5
            ) * math.inf

    metrics = Metrics(
        data,
        tokenizer=tokenizer,
    )
    print("Predictions:")
    metrics(
        SimpleNamespace(predictions=preds.cpu().numpy()),
        label_names_to_evaluate=["is_relevant"],
        max_n_visualize=None,
        visualize_out=f"predictions.{args.output_format}",
        use_categories=args.use_categories,
    )
    print("Labels (best possible metrics):")
    metrics(
        SimpleNamespace(predictions=labels.cpu().numpy()),
        label_names_to_evaluate=["is_relevant"],
        max_n_visualize=None,
        visualize_out=f"labels.{args.output_format}",
        use_categories=args.use_categories,
    )
