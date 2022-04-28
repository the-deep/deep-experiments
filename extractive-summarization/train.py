from bdb import Breakpoint
from calendar import c
import sys
import wandb
import math
import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from transformers.hf_argparser import HfArgumentParser
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm
from datasets import load_dataset
from functools import partial
from model import BasicModel, RecurrentModel
from rouge import Rouge
from typing import List


label_names = [
    "is_relevant",
    "has_sectors",
    "has_subpillars_1d",
    "has_subpillars_2d",
    "has_other",
]


@dataclass
class Args:
    model_name_or_path: str
    data_path: str
    excerpts_csv_path: str
    max_full_length: int
    max_length: int
    extra_context_length: int
    token_loss_weight: float = 1.0
    sentence_edit_threshold: int = math.inf
    n_subsample: int = None
    compute_relevant_with_or: bool = False
    loss_weights: List[float] = field(
        default_factory=lambda: [1.0] + [0.0] * (len(label_names) - 1)
    )


def get_label_vector(entry_id, excerpts_dict):
    label = np.zeros(len(label_names))

    for i, l in enumerate(label_names):
        label[i] = float(excerpts_dict[entry_id][l])

    return label


def encode(sample, tokenizer, args, excerpts_dict):
    sentences = sample["sentences"]
    text = sample["text"]

    assert sum(len(x) for x in sentences) == len(text)

    sentence_indices = [
        x["index"]
        for x in sample["excerpt_sentence_indices"]
        if x["distance"] < args.sentence_edit_threshold
    ]
    sentence_labels_list = [
        get_label_vector(x["source"], excerpts_dict)
        for x in sample["excerpt_sentence_indices"]
        if x["distance"] < args.sentence_edit_threshold
    ]

    encoding = tokenizer(
        sentences, add_special_tokens=False, return_offsets_mapping=True
    )

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
    token_labels = np.zeros((args.max_full_length, len(label_names)))

    for excerpt in sample["excerpts"]:
        e = excerpt["text"]
        e_source = excerpt["source"]

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
                label = get_label_vector(e_source, excerpts_dict)

                if i == 0 or not is_in_excerpt(offset_mapping[i - 1]):
                    # the token is at the boundary, could be encoded differently (i.e B- and I-)
                    # but currently encoded the same
                    token_labels[i] = label
                else:
                    token_labels[i] = label

    token_labels_mask = attention_mask.copy()
    token_labels_mask[np.where(input_ids == tokenizer.sep_token_id)] = 0
    token_labels_mask[np.where(input_ids == tokenizer.cls_token_id)] = 0

    # sentence labels
    sentence_labels = np.zeros((args.max_full_length, len(label_names)))
    sentence_labels_mask = np.zeros(args.max_full_length, dtype=np.int32)

    sep_positions = np.where(input_ids == tokenizer.sep_token_id)[0]
    sentence_indices_in_ids = [i for i in sentence_indices if i < len(sep_positions)]
    sentence_labels_in_ids = [
        label
        for i, label in zip(sentence_indices, sentence_labels_list)
        if i < len(sep_positions)
    ]

    if len(sentence_indices_in_ids) > 0:
        sentence_labels[sep_positions[sentence_indices_in_ids]] = sentence_labels_in_ids
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
    def __init__(self, dataset, tokenizer, args=None, training_args=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.scorer = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])

        self.called = False
        self.args = args
        self.training_args = training_args

    def compute_rouge(self, predicted_texts, true_texts):
        # bug in `rouge`: texts with only "." are treated as empty but not ignored
        evaluatable_pairs = [
            # lowercase because the model tokenizer might lowercase.
            # in that case it is not trivial to get uppercase predictions
            (hyp.lower(), ref.lower())
            for hyp, ref in zip(predicted_texts, true_texts)
            if len(hyp.replace(".", "")) > 0 and len(ref.replace(".", "")) > 0
        ]
        n_empty_hypotheses = sum(
            1
            for hyp, ref in zip(predicted_texts, true_texts)
            if (len(hyp.replace(".", "")) == 0) and len(ref.replace(".", "")) > 0
        )
        if len(evaluatable_pairs) == 0:
            scores = {}
        else:
            hyps, refs = list(zip(*evaluatable_pairs))
            all_scores = self.scorer.get_scores(
                hyps, refs, ignore_empty=True, avg=False
            )
            # flattens the dicts
            all_scores = [
                pd.json_normalize(score, sep="/").to_dict(orient="records")[0]
                for score in all_scores
            ]

            scores = {k: [0] * n_empty_hypotheses for k in all_scores[0].keys()}

            for score in all_scores:
                for key, value in score.items():
                    scores[key].append(value)

            scores = {k: np.mean(v) for k, v in scores.items()}

        return scores

    def get_metrics_for_label(
        self, results, label_index, threshold, max_n_visualize, visualize_out
    ):
        true_texts = [" ".join(sentences) for sentences in self.dataset["raw_excerpts"]]
        scores = {}

        ## sentence metrics
        predicted_indices = [
            np.where(
                (predictions[..., label_index] >= threshold)[np.array(labels_mask) == 1]
            )[0]
            for predictions, labels_mask in zip(
                results.predictions, self.dataset["sentence_labels_mask"]
            )
        ]
        true_indices = [
            np.where(
                (np.array(labels)[..., label_index] == 1)[np.array(labels_mask) == 1]
            )[0]
            for labels, labels_mask in zip(
                self.dataset["sentence_labels"], self.dataset["sentence_labels_mask"]
            )
        ]
        predicted_texts = [
            " ".join(sentences[i] for i in indices)
            for sentences, indices in zip(self.dataset["sentences"], predicted_indices)
        ]
        best_possible_predicted_texts = [
            " ".join(sentences[i] for i in indices)
            for sentences, indices in zip(self.dataset["sentences"], true_indices)
        ]

        flat_sentence_predictions = np.concatenate(
            [
                predictions[np.array(labels_mask).astype(bool), label_index]
                >= threshold
                for predictions, labels_mask in zip(
                    results.predictions, self.dataset["sentence_labels_mask"]
                )
            ]
        )
        flat_sentence_labels = np.concatenate(
            [
                np.array(labels)[np.array(mask).astype(bool), label_index]
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
        scores["sentence_recall"] = recall_score(
            flat_sentence_labels, flat_sentence_predictions
        )
        scores["sentence_precision"] = precision_score(
            flat_sentence_labels, flat_sentence_predictions
        )

        ## token predictions
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
            self.dataset["token_labels"],
        ):
            attention_mask = np.array(attention_mask).astype(bool)
            labels = np.array(labels)
            input_ids = np.array(input_ids)
            offset_mapping = np.array(offset_mapping)

            mask = attention_mask & ((predictions[..., label_index] >= threshold))
            best_mask = attention_mask & ((labels[..., label_index] == 1))

            predicted_texts.append(
                self.tokenizer.decode(input_ids[mask], skip_special_tokens=True)
            )
            best_possible_predicted_texts.append(
                self.tokenizer.decode(input_ids[best_mask], skip_special_tokens=True)
            )

        flat_predictions = np.concatenate(
            [
                predictions[np.array(labels_mask).astype(bool), label_index]
                >= threshold
                for predictions, labels_mask in zip(
                    token_predictions, self.dataset["token_labels_mask"]
                )
            ]
        )

        flat_labels = np.concatenate(
            [
                np.array(labels)[np.array(mask).astype(bool), label_index]
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

        scores["token_accuracy"] = accuracy_score(flat_labels, flat_predictions)
        scores["token_balanced_accuracy"] = balanced_accuracy_score(
            flat_labels, flat_predictions
        )
        scores["token_f1_score"] = f1_score(flat_labels, flat_predictions)
        scores["token_recall"] = recall_score(flat_labels, flat_predictions)
        scores["token_precision"] = precision_score(flat_labels, flat_predictions)

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
            if max_n_visualize is not None and i >= max_n_visualize:
                break

            probs = predictions[..., label_index]

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
        if self.training_args is not None and "wandb" in self.training_args.report_to:
            wandb.log(
                {f"highlighted_texts_{label_names[label_index]}": wandb.Html(html)}
            )
        if visualize_out is not None:
            open(visualize_out, "w").write(html)

        print("METRICS:", scores)

        return scores

    def __call__(self, results, threshold=0.5, max_n_visualize=10, visualize_out=None):
        all_scores = {}

        results.predictions[:] = torch.sigmoid(
            torch.from_numpy(results.predictions).float()
        ).numpy()

        # only compute metrics for relevancy since it takes quite some time
        for label_name in label_names:
            idx = label_names.index(label_name)

            if (
                label_name == "is_relevant"
                and self.args is not None
                and self.args.compute_relevant_with_or
            ):
                other_idx = sorted(set(np.arange(len(label_names))) - {idx})
                results.predictions[..., idx] = results.predictions[..., other_idx].max(
                    axis=-1
                )

            scores = self.get_metrics_for_label(
                results,
                idx,
                threshold,
                max_n_visualize,
                visualize_out,
            )

            all_scores.update({f"{label_name}/{k}": v for k, v in scores.items()})

        self.called = True

        return all_scores


def train(args, training_args):
    data = load_dataset(
        "json",
        data_files=args.data_path,
        split="train" if args.n_subsample is None else f"train[:{args.n_subsample}]",
    )
    data = data.filter(lambda x: len(x["excerpts"]) > 0)

    if os.path.exists("excerpts_dict.pkl"):
        with open("excerpts_dict.pkl", "rb") as f:
            excerpts_dict = pickle.load(f)
    else:
        excerpts_df = pd.read_csv(args.excerpts_csv_path)
        excerpts_df["has_sectors"] = excerpts_df["sectors"].apply(eval).apply(len) > 0
        excerpts_df["has_subpillars_1d"] = (
            excerpts_df["subpillars_1d"].apply(eval).apply(len) > 0
        )
        excerpts_df["has_subpillars_2d"] = (
            excerpts_df["subpillars_2d"].apply(eval).apply(len) > 0
        )
        excerpts_df["is_relevant"] = 1
        excerpts_df["has_other"] = ~(
            excerpts_df["has_sectors"]
            | excerpts_df["has_subpillars_1d"]
            | excerpts_df["has_subpillars_2d"]
        )
        excerpts_dict = {}
        for _, row in excerpts_df.iterrows():
            excerpts_dict[row["entry_id"]] = {
                k: v for k, v in row.to_dict().items() if k in label_names
            }
        pickle.dump(excerpts_dict, open("excerpts_dict.pkl", "wb"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    data = data.map(
        partial(encode, tokenizer=tokenizer, args=args, excerpts_dict=excerpts_dict),
        num_proc=training_args.dataloader_num_workers,
    )

    datasets = {}
    train_indices, eval_indices = train_test_split(
        np.arange(len(data)), test_size=0.01, shuffle=True, random_state=1234
    )
    datasets["train"] = data.select(train_indices)
    datasets["test"] = data.select(eval_indices)

    model = BasicModel(
        args.model_name_or_path,
        tokenizer,
        num_labels=len(label_names),
        token_loss_weight=args.token_loss_weight,
        loss_weights=args.loss_weights,
        slice_length=args.max_length,
        extra_context_length=args.extra_context_length,
    )

    if "wandb" in training_args.report_to and training_args.do_train:
        wandb.init(project="deep")
        wandb.config.update(training_args)
        wandb.config.update(args)
        wandb.save(__file__, policy="now")
    else:
        training_args.report_to = []

    training_args.label_names = ["token_labels", "sentence_labels"]

    metrics = Metrics(
        datasets["test"], tokenizer=tokenizer, training_args=training_args, args=args
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
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
