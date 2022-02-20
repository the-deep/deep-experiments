import sys
import json
import wandb
import math

import numpy as np
import pandas as pd
from dataclasses import dataclass
from transformers.hf_argparser import HfArgumentParser
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm.auto import tqdm
from datasets import load_dataset
from functools import partial
from model import BasicModel, RecurrentModel
from rouge import Rouge


@dataclass
class Args:
    model_name_or_path: str
    data_path: str
    mode: str = "sentence"
    max_full_length: int = 4096
    max_length: int = 512
    n_subsample: int = None


O_LABEL = 0
I_LABEL = 1
B_LABEL = 1


def encode_as_sentences(sample, tokenizer, args):
    sentences = sample["sentences"]
    sentence_indices = [x["index"] for x in sample["sentence_indices"] if x["distance"] < math.inf]

    if len(sentences) > 0:
        sentence_input_ids = tokenizer(sentences, add_special_tokens=False)["input_ids"]
    else:
        # TODO: empty texts should be removed in preprocessing
        sentence_input_ids = []
    input_ids = [tokenizer.cls_token_id]

    for sentence_ids in sentence_input_ids:
        input_ids.extend(sentence_ids)
        input_ids.append(tokenizer.sep_token_id)

        if len(input_ids) >= args.max_full_length:
            input_ids = input_ids[: args.max_full_length]
            break

    while len(input_ids) < args.max_full_length:
        input_ids.append(tokenizer.pad_token_id)

    input_ids = np.array(input_ids, dtype=np.int32)
    labels = np.zeros(args.max_full_length, dtype=np.int32)
    labels_mask = np.zeros(args.max_full_length, dtype=np.int32)
    attention_mask = np.zeros(args.max_full_length, dtype=np.int32)

    sep_positions = np.where(input_ids == tokenizer.sep_token_id)[0]
    sentence_indices_in_ids = [i for i in sentence_indices if i < len(sep_positions)]

    labels[sep_positions[sentence_indices_in_ids]] = B_LABEL
    labels_mask[sep_positions] = 1
    attention_mask[np.where(input_ids != tokenizer.pad_token_id)] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "labels_mask": labels_mask,
    }


def encode_as_tokens(sample, tokenizer, args):
    text, excerpt = sample["text"], sample["excerpt"]

    n_slices = args.max_full_length // args.max_length

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        max_length=args.max_full_length - n_slices,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    offset_mapping = encoding.offset_mapping

    labels = torch.full((args.max_full_length - n_slices,), O_LABEL, dtype=torch.long)

    for e in excerpt:
        start_index = text.index(e)
        end_index = start_index + len(e)

        def is_in_excerpt(offset):
            return offset[0] != offset[1] and offset[0] >= start_index and offset[1] <= end_index

        for i, offset in enumerate(offset_mapping):
            if is_in_excerpt(offset):
                if i == 0 or not is_in_excerpt(offset_mapping[i - 1]):
                    labels[i] = B_LABEL
                else:
                    labels[i] = I_LABEL

    input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)
    offset_mapping = torch.tensor(encoding["offset_mapping"], dtype=torch.long)

    input_ids = torch.cat(
        [
            torch.full((n_slices, 1), tokenizer.cls_token_id, dtype=torch.long),
            input_ids.view((n_slices, -1)),
        ],
        1,
    ).view(args.max_full_length)
    attention_mask = torch.cat(
        [
            torch.ones((n_slices, 1), dtype=torch.long),
            attention_mask.view((n_slices, -1)),
        ],
        1,
    ).view(args.max_full_length)
    offset_mapping = torch.cat(
        [
            torch.zeros((n_slices, 1, 2), dtype=torch.long),
            offset_mapping.view((n_slices, -1, 2)),
        ],
        1,
    ).view(args.max_full_length, 2)
    labels = torch.cat(
        [
            torch.zeros((n_slices, 1), dtype=torch.long),
            labels.view((n_slices, -1)),
        ],
        1,
    ).view(args.max_full_length)

    return {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "offset_mapping": offset_mapping.numpy(),
        "labels": labels.numpy(),
        "labels_mask": attention_mask.numpy(),
    }


class Metrics:
    def __init__(self, dataset, tokenizer, mode):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.scorer = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])

        self.called = False
        self.mode = mode

    def compute_rouge(self, predicted_texts, true_texts):
        # bug in `rouge`: texts with only "." are treated as empty but not ignored
        evaluatable_pairs = [
            (hyp, ref)
            for hyp, ref in zip(predicted_texts, true_texts)
            if len(hyp.replace(".", "")) > 0 and len(ref.replace(".", "")) > 0
        ]
        if len(evaluatable_pairs) == 0:
            scores = {}
        else:
            hyps, refs = list(zip(*evaluatable_pairs))
            scores = self.scorer.get_scores(hyps, refs, ignore_empty=True, avg=True)
            # flattens the dict
            scores = pd.json_normalize(scores, sep="/").to_dict(orient="records")[0]

        return scores

    def __call__(self, results):
        true_texts = [" ".join(sentences) for sentences in self.dataset["excerpt"]]

        if self.mode == "sentence":
            predicted_indices = [
                np.where((predictions.argmax(-1) == B_LABEL)[np.array(labels_mask) == 1])[0]
                for predictions, labels_mask in zip(
                    results.predictions, self.dataset["labels_mask"]
                )
            ]
            true_indices = [
                np.where((np.array(labels) == B_LABEL)[np.array(labels_mask) == 1])[0]
                for labels, labels_mask in zip(self.dataset["labels"], self.dataset["labels_mask"])
            ]
            predicted_texts = [
                " ".join(sentences[i] for i in indices)
                for sentences, indices in zip(self.dataset["sentences"], predicted_indices)
            ]
            best_possible_predicted_texts = [
                " ".join(sentences[i] for i in indices)
                for sentences, indices in zip(self.dataset["sentences"], true_indices)
            ]
        else:
            predicted_texts = []
            best_possible_predicted_texts = []

            for (text, input_ids, attention_mask, offset_mapping, predictions, labels,) in zip(
                self.dataset["text"],
                self.dataset["input_ids"],
                self.dataset["attention_mask"],
                self.dataset["offset_mapping"],
                results.predictions,
                self.dataset["labels"],
            ):
                attention_mask = np.array(attention_mask).astype(bool)
                labels = np.array(labels)
                input_ids = np.array(input_ids)
                offset_mapping = np.array(offset_mapping)

                mask = attention_mask & (
                    (predictions.argmax(-1) == B_LABEL) | (predictions.argmax(-1) == I_LABEL)
                )
                best_mask = attention_mask & ((labels == B_LABEL) | (labels == I_LABEL))

                predicted_texts.append(
                    "".join(
                        # TODO: maybe there's a way to make this independent of the tokenizer
                        ("" if self.tokenizer.decode(i).startswith("##") else " ") + text[start:end]
                        for i, (start, end) in zip(input_ids[mask], offset_mapping[mask])
                    )
                )
                best_possible_predicted_texts.append(
                    "".join(
                        ("" if self.tokenizer.decode(i).startswith("##") else " ") + text[start:end]
                        for i, (start, end) in zip(input_ids[best_mask], offset_mapping[best_mask])
                    )
                )

        scores = {}
        scores.update(self.compute_rouge(predicted_texts, true_texts))

        flat_predictions = np.concatenate(
            [
                predictions[np.array(labels_mask).astype(bool)].argmax(-1)
                for predictions, labels_mask in zip(
                    results.predictions, self.dataset["labels_mask"]
                )
            ]
        )
        flat_labels = np.concatenate(
            [
                labels[np.array(labels_mask).astype(bool)]
                for labels, labels_mask in zip(results.label_ids, self.dataset["labels_mask"])
            ]
        )

        if not self.called:
            scores.update(
                {
                    "best_" + k: v
                    for k, v in self.compute_rouge(
                        best_possible_predicted_texts, true_texts
                    ).items()
                }
            )
            scores["unit_baseline_accuracy"] = accuracy_score(
                flat_labels, np.zeros_like(flat_labels)
            )
            self.called = True

        scores["unit_accuracy"] = accuracy_score(flat_labels, flat_predictions)
        scores["unit_balanced_accuracy"] = balanced_accuracy_score(flat_labels, flat_predictions)
        scores["unit_f1_score"] = f1_score(flat_labels, flat_predictions)

        return scores


def train(args, training_args):
    data = load_dataset(
        "json",
        data_files=args.data_path,
        split="train" if args.n_subsample is None else f"train[:{args.n_subsample}]",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    encode_fn = encode_as_sentences if args.mode == "sentence" else encode_as_tokens

    data = data.map(
        partial(encode_fn, tokenizer=tokenizer, args=args),
        num_proc=training_args.dataloader_num_workers,
    )
    data = data.train_test_split(test_size=0.1, shuffle=True, seed=1234)

    model = BasicModel(args.model_name_or_path, num_labels=2, slice_length=args.max_length)

    if "wandb" in training_args.report_to:
        wandb.init(project="deep")
        wandb.config.update(training_args)
        wandb.config.update(args)
        wandb.save(__file__, policy="now")

    metrics = Metrics(data["test"], tokenizer=tokenizer, mode=args.mode)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        compute_metrics=metrics,
    )
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser([Args, TrainingArguments])

    (args, training_args) = parser.parse_json_file(sys.argv[1])
    train(args, training_args)
