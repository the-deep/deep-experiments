import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


LABEL_NAMES = [
    "is_relevant",
    "has_sectors",
    "has_subpillars_1d",
    "has_subpillars_2d",
    "has_other",
    "has_sector_Agriculture",
    "has_sector_Cross",
    "has_sector_Education",
    "has_sector_Food Security",
    "has_sector_Health",
    "has_sector_Livelihoods",
    "has_sector_Logistics",
    "has_sector_NOT_MAPPED",
    "has_sector_Nutrition",
    "has_sector_Protection",
    "has_sector_Shelter",
    "has_sector_WASH",
]


class ExtractionDataset(Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, idx):
        d = self.dset[idx]
        return {
            "id": d["id"],
            "input_ids": d["input_ids"],
            "attention_mask": d["attention_mask"],
            "offset_mapping": d["offset_mapping"],
            "token_labels": d["token_labels"],
            "token_labels_mask": d["token_labels_mask"],
            "sentence_labels": d["sentence_labels"],
            "sentence_labels_mask": d["sentence_labels_mask"],
        }

    def __len__(self):
        return len(self.dset)


class log_time:
    def __init__(self, name="BLOCK", extra_args=None):
        self.name = name
        self.extra_args = extra_args
        self.start = 0
        self.end = 0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args, **kwargs):
        self.end = time.time()
        diff = self.end - self.start
        print("LOG TIME:", f"'{self.name}' took {round(diff, 5)}seconds to run!")


def create_excerpts_dict(excerpts_df):
    possible_sectors = set(s for sectors in excerpts_df["sectors"].values for s in eval(sectors))

    has_sectors = np.zeros(len(excerpts_df), dtype=bool)
    sectors_value_dict = {s: np.zeros_like(has_sectors) for s in possible_sectors}

    for i, row in excerpts_df.iterrows():
        for sector in eval(row["sectors"]):
            sectors_value_dict[sector][i] = 1
            has_sectors[i] = True

    excerpts_df["has_sectors"] = has_sectors

    for key, value in sectors_value_dict.items():
        excerpts_df[f"has_sector_{key}"] = value

    excerpts_df["has_subpillars_1d"] = (
        excerpts_df["subpillars_1d"].apply(eval).apply(len) > 0
        if "subpillars_1d" in excerpts_df
        else False
    )
    excerpts_df["has_subpillars_2d"] = (
        excerpts_df["subpillars_2d"].apply(eval).apply(len) > 0
        if "subpillars_2d" in excerpts_df
        else False
    )
    excerpts_df["is_relevant"] = 1
    excerpts_df["has_other"] = ~(
        excerpts_df["has_sectors"]
        | excerpts_df["has_subpillars_1d"]
        | excerpts_df["has_subpillars_2d"]
    )

    for key in LABEL_NAMES:
        if key not in excerpts_df.columns:
            print(f"Warning: {key} not present in excerpts dataframe. Filling with 'false'.")
            excerpts_df[key] = False

    excerpts_dict = {}
    for _, row in excerpts_df.iterrows():
        excerpts_dict[row["entry_id"]] = {k: v for k, v in row.iteritems() if k in LABEL_NAMES}

    return excerpts_dict


def get_label_vector(entry_id, excerpts_dict=None):
    label = np.zeros(len(LABEL_NAMES))
    # every excerpt is relevant
    label[LABEL_NAMES.index("is_relevant")] = 1

    if excerpts_dict is not None:
        for i, l in enumerate(LABEL_NAMES):
            label[i] = float(excerpts_dict[entry_id][l])

    return label


def encode(sample, tokenizer, args, excerpts_dict=None):
    sentences = sample["sentences"]
    text = sample["text"]

    has_labels = "excerpts" in sample

    encoding = tokenizer(sentences, add_special_tokens=False, return_offsets_mapping=True)

    input_ids = [tokenizer.cls_token_id]
    offset_mapping = [(0, 0)]

    prev_offset = 0

    for (sentence, sentence_ids, sentence_offsets) in zip(
        sentences, encoding["input_ids"], encoding["offset_mapping"]
    ):
        input_ids.extend(sentence_ids)
        input_ids.append(tokenizer.sep_token_id)

        offset_mapping.extend(
            [(prev_offset + start, prev_offset + end) for start, end in sentence_offsets]
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
    if has_labels:
        token_labels = np.zeros((args.max_full_length, len(LABEL_NAMES)))

        for excerpt in sample["excerpts"]:
            e = excerpt["text"]
            e_source = excerpt["source"]

            start_index = text.index(e)
            end_index = start_index + len(e)

            def is_in_excerpt(offset):
                return (
                    offset[0] != offset[1] and offset[0] >= start_index and offset[1] <= end_index
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
    else:
        token_labels = token_labels_mask = None

    # sentence labels
    if has_labels and sentences is not None:
        assert sum(len(x) for x in sentences) == len(text)

        sentence_indices = (
            [
                x["index"]
                for x in sample["excerpt_sentence_indices"]
                if x["distance"] < args.sentence_edit_threshold
            ]
            if has_labels
            else None
        )
        sentence_labels_list = (
            [
                get_label_vector(x["source"], excerpts_dict)
                for x in sample["excerpt_sentence_indices"]
                if x["distance"] < args.sentence_edit_threshold
            ]
            if has_labels
            else None
        )

        sentence_labels = np.zeros((args.max_full_length, len(LABEL_NAMES)))
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
    else:
        sentence_labels = sentence_labels_mask = None

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
    }

    if has_labels:
        out.update(
            {
                "token_labels": token_labels,
                "token_labels_mask": token_labels_mask,
                "sentence_labels": sentence_labels,
                "sentence_labels_mask": sentence_labels_mask,
            }
        )

    return out


def get_test_train_data(args, training_args):
    with log_time("load_dataset and filter excerpts"):
        data = load_dataset(
            "json",
            data_files=args.data_path,
            split="train" if args.n_subsample is None else f"train[:{args.n_subsample}]",
        )
        data = data.filter(lambda x: len(x["excerpts"]) > 0)

    excerpts_df = pd.read_csv(args.excerpts_csv_path)
    with log_time("create excerpts dict"):
        excerpts_dict = create_excerpts_dict(excerpts_df)

    with log_time("tokenizer construction"):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    with log_time("map encode to excerpts"):
        data = data.map(
            partial(encode, tokenizer=tokenizer, args=args, excerpts_dict=excerpts_dict),
            num_proc=training_args.dataloader_num_workers,
        )

    train_indices_, eval_indices = train_test_split(
        np.arange(len(data)), test_size=0.01, shuffle=True, random_state=1234
    )
    train_indices, val_indices = train_test_split(train_indices_, test_size=0.1, random_state=1234)
    train = data.select(train_indices)  # list of <something>
    test = data.select(eval_indices)
    val = data.select(val_indices)
    return train, test, val
