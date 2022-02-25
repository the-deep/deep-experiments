from bdb import Breakpoint
from calendar import c
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


def encode(sample, tokenizer, args):
    sentences = sample["sentences"]
    text = sample["text"]

    assert sum(len(x) for x in sentences) == len(text)

    sentence_indices = [
        x["index"]
        for x in sample["excerpt_sentence_indices"]
        if x["distance"] < math.inf
    ]

    if len(sentences) > 0:
        encoding = tokenizer(
            sentences, add_special_tokens=False, return_offsets_mapping=True
        )
    else:
        # TODO: empty texts should be removed in preprocessing
        encoding = {"input_ids": [], "offset_mapping": []}
    input_ids = [tokenizer.cls_token_id]
    offset_mapping = [(0, 0)]

    prev_offset = 0

    for (sentence, sentence_ids, sentence_offsets) in zip(
        sentences, encoding["input_ids"], encoding["offset_mapping"]
    ):
        input_ids.extend(sentence_ids)
        input_ids.append(tokenizer.sep_token_id)

        offset_mapping.extend(
            [
                (prev_offset + start, prev_offset + end)
                for start, end in sentence_offsets
            ]
        )
        offset_mapping.append((0, 0))

        prev_offset += len(sentence)

        if len(input_ids) >= args.max_full_length:
            input_ids = input_ids[: args.max_full_length]
            offset_mapping = offset_mapping[: args.max_full_length]
            break

    while len(input_ids) < args.max_full_length:
        input_ids.append(tokenizer.pad_token_id)
        offset_mapping.append((0, 0))

    input_ids = np.array(input_ids, dtype=np.int32)
    offset_mapping = np.array(offset_mapping, dtype=np.int32)

    attention_mask = np.zeros(args.max_full_length, dtype=np.int32)
    attention_mask[np.where(input_ids != tokenizer.pad_token_id)] = 1

    # token labels
    token_labels = torch.full((args.max_full_length,), O_LABEL, dtype=torch.long)

    for e in sample["excerpts"]:
        start_index = text.index(e)
        end_index = start_index + len(e)

        def is_in_excerpt(offset):
            return (
                offset[0] != offset[1]
                and offset[0] >= start_index
                and offset[1] <= end_index
            )

        for i, offset in enumerate(offset_mapping):
            if is_in_excerpt(offset):
                if i == 0 or not is_in_excerpt(offset_mapping[i - 1]):
                    token_labels[i] = I_LABEL
                else:
                    token_labels[i] = I_LABEL

    token_labels_mask = attention_mask.copy()
    token_labels_mask[np.where(input_ids == tokenizer.sep_token_id)] = 0
    token_labels_mask[np.where(input_ids == tokenizer.cls_token_id)] = 0

    # sentence labels
    sentence_labels = np.zeros(args.max_full_length, dtype=np.int32)
    sentence_labels_mask = np.zeros(args.max_full_length, dtype=np.int32)

    sep_positions = np.where(input_ids == tokenizer.sep_token_id)[0]
    sentence_indices_in_ids = [i for i in sentence_indices if i < len(sep_positions)]

    sentence_labels[sep_positions[sentence_indices_in_ids]] = I_LABEL
    sentence_labels_mask[sep_positions] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_labels": token_labels,
        "token_labels_mask": token_labels_mask,
        "sentence_labels": sentence_labels,
        "sentence_labels_mask": sentence_labels_mask,
        "offset_mapping": offset_mapping,
    }


class Metrics:
    def __init__(self, dataset, tokenizer, mode, training_args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.scorer = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])

        self.called = False
        self.mode = mode
        self.training_args = training_args

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
        true_texts = [" ".join(sentences) for sentences in self.dataset["raw_excerpts"]]
        scores = {}

        if self.mode == "sentence":
            token_predictions = np.zeros(
                (len(self.dataset), len(self.dataset["input_ids"][0]), 2),
                dtype=np.int32,
            )

            predicted_indices = [
                np.where(
                    (predictions.argmax(-1) == I_LABEL)[np.array(labels_mask) == 1]
                )[0]
                for predictions, labels_mask in zip(
                    results.predictions, self.dataset["labels_mask"]
                )
            ]

            for i, (predictions, labels_mask, labels) in enumerate(
                zip(
                    results.predictions,
                    self.dataset["labels_mask"],
                    self.dataset["labels"],
                )
            ):
                prev_j = 1  # zero is cls token
                for j in np.where(labels_mask)[0]:
                    token_predictions[i, prev_j:j] = predictions[j]
                    prev_j = j + 1  # skip sep token

            true_indices = [
                np.where((np.array(labels) == I_LABEL)[np.array(labels_mask) == 1])[0]
                for labels, labels_mask in zip(
                    self.dataset["labels"], self.dataset["labels_mask"]
                )
            ]
            predicted_texts = [
                " ".join(sentences[i] for i in indices)
                for sentences, indices in zip(
                    self.dataset["sentences"], predicted_indices
                )
            ]
            best_possible_predicted_texts = [
                " ".join(sentences[i] for i in indices)
                for sentences, indices in zip(self.dataset["sentences"], true_indices)
            ]

            flat_sentence_predictions = np.concatenate(
                [
                    predictions[np.array(labels_mask).astype(bool)].argmax(-1)
                    for predictions, labels_mask in zip(
                        results.predictions, self.dataset["sentence_labels_mask"]
                    )
                ]
            )
            flat_sentence_labels = np.concatenate(
                [
                    np.array(labels)[np.array(mask).astype(bool)]
                    for labels, mask in zip(
                        self.dataset["sentence_labels"],
                        self.dataset["sentence_labels_mask"],
                    )
                ]
            )

            if not self.called:
                scores["sentence_baseline_accuracy"] = accuracy_score(
                    flat_sentence_labels, np.zeros_like(flat_sentence_labels)
                )

            scores["sentence_accuracy"] = accuracy_score(
                flat_sentence_labels, flat_sentence_predictions
            )
            scores["sentence_balanced_accuracy"] = balanced_accuracy_score(
                flat_sentence_labels, flat_sentence_predictions
            )
            scores["sentence_f1_score"] = f1_score(
                flat_sentence_labels, flat_sentence_predictions
            )
        else:
            token_predictions = results.predictions

            predicted_texts = []
            best_possible_predicted_texts = []

            for (
                text,
                input_ids,
                attention_mask,
                offset_mapping,
                predictions,
                labels,
            ) in zip(
                self.dataset["text"],
                self.dataset["input_ids"],
                self.dataset["attention_mask"],
                self.dataset["offset_mapping"],
                token_predictions,
                self.dataset["labels"],
            ):
                attention_mask = np.array(attention_mask).astype(bool)
                labels = np.array(labels)
                input_ids = np.array(input_ids)
                offset_mapping = np.array(offset_mapping)

                mask = attention_mask & ((predictions.argmax(-1) == I_LABEL))
                best_mask = attention_mask & ((labels == I_LABEL) | (labels == I_LABEL))

                predicted_texts.append(
                    "".join(
                        # TODO: maybe there's a way to make this independent of the tokenizer
                        ("" if self.tokenizer.decode(i).startswith("##") else " ")
                        + text[start:end]
                        for i, (start, end) in zip(
                            input_ids[mask], offset_mapping[mask]
                        )
                    )
                )
                best_possible_predicted_texts.append(
                    "".join(
                        ("" if self.tokenizer.decode(i).startswith("##") else " ")
                        + text[start:end]
                        for i, (start, end) in zip(
                            input_ids[best_mask], offset_mapping[best_mask]
                        )
                    )
                )

        flat_predictions = np.concatenate(
            [
                predictions[np.array(labels_mask).astype(bool)].argmax(-1)
                for predictions, labels_mask in zip(
                    token_predictions, self.dataset["token_labels_mask"]
                )
            ]
        )

        flat_labels = np.concatenate(
            [
                np.array(labels)[np.array(mask).astype(bool)]
                for labels, mask in zip(
                    self.dataset["token_labels"], self.dataset["token_labels_mask"]
                )
            ]
        )

        scores.update(self.compute_rouge(predicted_texts, true_texts))

        if not self.called:
            scores.update(
                {
                    "best_" + k: v
                    for k, v in self.compute_rouge(
                        best_possible_predicted_texts, true_texts
                    ).items()
                }
            )
            scores["token_baseline_accuracy"] = accuracy_score(
                flat_labels, np.zeros_like(flat_labels)
            )
            self.called = True

        scores["token_accuracy"] = accuracy_score(flat_labels, flat_predictions)
        scores["token_balanced_accuracy"] = balanced_accuracy_score(
            flat_labels, flat_predictions
        )
        scores["token_f1_score"] = f1_score(flat_labels, flat_predictions)

        html = ""

        for i, (id, text, predictions, offset_mapping) in enumerate(
            zip(
                self.dataset["id"],
                self.dataset["text"],
                token_predictions,
                self.dataset["offset_mapping"],
            )
        ):
            # otherwise HTML gets too large
            if i >= 10:
                break

            probs = torch.softmax(torch.from_numpy(predictions).to(torch.float32), -1)[
                :, I_LABEL
            ]

            html += f"<h2>Text #{i} - ID {id}</h2>"
            html += "<p>"

            prev_end = 0

            for (start, end), prob in zip(offset_mapping, probs):
                if start == end:
                    continue

                # include preceding whitespace
                while start > 0 and text[start - 1].isspace():
                    start -= 1

                html += text[prev_end:start]
                if prob > 0.1:
                    html += f'<span style="background-color:rgba(255, 0, 0, {prob});">{text[start:end]}</span>'
                else:
                    html += text[start:end]

                prev_end = end

            html += "/<p>"

        html = html.replace("\n", "<br>")
        if "wandb" in self.training_args.report_to:
            wandb.log({"highlighted_texts": wandb.Html(html)})

        print("METRICS:", scores)

        return scores


def train(args, training_args):
    data = load_dataset(
        "json",
        data_files=args.data_path,
        split="train" if args.n_subsample is None else f"train[:{args.n_subsample}]",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    data = data.map(
        partial(encode, tokenizer=tokenizer, args=args),
        num_proc=training_args.dataloader_num_workers,
    )
    data = data.add_column("labels", data[f"{args.mode}_labels"])
    data = data.add_column("labels_mask", data[f"{args.mode}_labels_mask"])

    data = data.train_test_split(test_size=0.1, shuffle=True, seed=1234)

    model = BasicModel(
        args.model_name_or_path, num_labels=2, slice_length=args.max_length
    )
    # model.load_state_dict(torch.load("output/checkpoint-960/pytorch_model.bin"))

    if "wandb" in training_args.report_to and training_args.do_train:
        wandb.init(project="deep")
        wandb.config.update(training_args)
        wandb.config.update(args)
        wandb.save(__file__, policy="now")
    else:
        training_args.report_to = []

    metrics = Metrics(
        data["test"], tokenizer=tokenizer, mode=args.mode, training_args=training_args
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        compute_metrics=metrics,
    )

    if training_args.do_train:
        trainer.train()
    elif training_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    parser = HfArgumentParser([Args, TrainingArguments])

    (args, training_args) = parser.parse_json_file(sys.argv[1])
    train(args, training_args)
