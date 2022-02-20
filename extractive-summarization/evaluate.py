import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np

from transformers.hf_argparser import HfArgumentParser

from model import RecurrentModel


@dataclass
class Args:
    model_name_or_path: str
    data_path: str
    max_full_length: int = 4096
    max_length: int = 256
    threshold: float = 0.5


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
    model = RecurrentModel(
        "microsoft/xtremedistil-l6-h256-uncased",
        num_labels=3,
        slice_length=args.max_length,
    )
    model.load_state_dict(torch.load(args.model_name_or_path))

    data = load_dataset("json", data_files=args.data_path, split="train")
    sample = data[0]

    encoding = tokenizer(
        sample["text"],
        return_offsets_mapping=True,
        return_tensors="pt",
        max_length=args.max_full_length,
        truncation=True,
        padding=True,
    )

    print("** Text **")
    print(sample["text"])
    print("** Excerpts **")
    pprint(sample["excerpt"])

    with torch.inference_mode():
        logits = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"],)[
            "logits"
        ][0]

    probs = torch.softmax(logits, -1)
    mask = probs.argmax(-1)

    plt.plot(probs[:, 0])
    plt.plot(probs[:, 1])
    plt.ylim([0, 1])
    plt.show()

    output = tokenizer.decode(
        encoding["input_ids"][0][encoding["attention_mask"][0].to(bool) & (mask > 0)],
        skip_special_tokens=True,
    )
    print("** Predicted Text **")
    print(output)
