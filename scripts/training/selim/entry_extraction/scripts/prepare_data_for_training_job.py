from functools import partial
import time
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset
from utils import flatten, process_tag, prepare_X_data
from ast import literal_eval
import torch
import json


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


class DataPreparation:
    def __init__(
        self,
        raw_data_path: str,
        excerpts_df_path: pd.DataFrame,
        tokenizer_name_or_path: str,
        dataloader_num_workers: int = 1,
    ):
        """
        need to ensure that the tokenizer used here is the same as the one used in the models.
        """
        self.dataloader_num_workers = dataloader_num_workers

        self.original_data = load_dataset(
            "json",
            data_files=raw_data_path,
            split="train",
        )
        self.original_data = self.original_data.filter(lambda x: len(x["excerpts"]) > 0)

        self.excerpts_df = pd.read_csv(excerpts_df_path)[
            ["entry_id", "sectors", "subpillars_1d", "subpillars_2d", "lead_id"]
        ]
        self.lead_ids = self.excerpts_df.lead_id.unique().tolist()
        self._create_excerpts_dict()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def _create_excerpts_dict(self):

        self.excerpts_df["primary_tags"] = self.excerpts_df.apply(
            lambda x: flatten(
                [
                    process_tag(literal_eval(x[tag]), tag)
                    for tag in ["sectors", "subpillars_1d", "subpillars_2d"]
                ]
            ),
            axis=1,
        )

        self.excerpts_df["primary_tags"] = self.excerpts_df["primary_tags"].apply(
            lambda x: x + ["is_relevant"] if len(x) > 0 else x
        )

        self.excerpts_dict = dict(
            zip(self.excerpts_df.entry_id, self.excerpts_df.primary_tags)
        )

    def _get_label_vector(self, entry_id: int):
        target_ids = torch.zeros(len(self.tagname_to_tagid), dtype=torch.long)

        # every excerpt is relevant
        target_ids[self.tagname_to_tagid["is_relevant"]] = 1

        if self.excerpts_dict is not None:
            entry_tags: List = self.excerpts_dict[entry_id]
            entry_tag_ids = [self.tagname_to_tagid[tag] for tag in entry_tags]
            target_ids[entry_tag_ids] = 1

        return target_ids

    def _create_y_data(self, sample, input_ids, attention_mask, offset_mapping):
        # TODO: recheck all works

        self.label_names = sorted(list(set(flatten(self.excerpts_dict.values()))))
        self.tagname_to_tagid = {
            tag_name: tag_id for tag_id, tag_name in enumerate(self.label_names)
        }

        token_labels = torch.zeros(
            (input_ids.shape[0], len(self.label_names)), dtype=torch.long
        )
        text = sample["text"]

        for excerpt in sample["excerpts"]:
            e = excerpt["text"]
            entry_id = excerpt["source"]

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
                    label = self._get_label_vector(entry_id)

                    # TODO: understand and fix this
                    if i == 0 or not is_in_excerpt(offset_mapping[i - 1]):
                        # the token is at the boundary, could be encoded differently (i.e B- and I-)
                        # but currently encoded the same
                        token_labels[i] = label
                    else:
                        token_labels[i] = label

        token_labels_mask = attention_mask.clone()
        token_labels_mask[torch.where(input_ids == self.tokenizer.sep_token_id)] = 0
        token_labels_mask[torch.where(input_ids == self.tokenizer.cls_token_id)] = 0

        return (token_labels, token_labels_mask)

    def _encode(self, sample):

        input_ids, attention_mask, offset_mapping = prepare_X_data(
            sample["sentences"], self.tokenizer
        )

        token_has_labels = "excerpts" in sample
        if token_has_labels:
            token_labels, token_labels_mask = self._create_y_data(
                sample, input_ids, attention_mask, offset_mapping
            )
        else:
            token_labels = token_labels_mask = None

        out = {
            "lead_in_classification_data": True,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "token_labels": token_labels,
            "token_labels_mask": token_labels_mask,
        }

        return out

    def __call__(self, testing: bool = False, save_split_dicts: bool = True):

        """
        testing: bool for whether we are testing the ddf or if we want all the data
        save_split_dicts: bool: whether or not to save the data split to train val test

        - preprocessing function, to be run before training models
        - output is raw
            - no max length
            - still need to add context and to pad etc to lengths, which is done in the training,
                because different cdepending on the context we want to add

        """

        # sanity check: make sur all leads are in excerpts df.
        used_data = self.original_data
        ids = used_data["id"]
        selected_ids = [
            i
            for i, lead_proj_ids in enumerate(ids)
            if lead_proj_ids[0] in self.lead_ids
        ]
        used_data = used_data.select(selected_ids)

        # testing: make sure everything is working
        if testing:
            n_sample_rows = 100
            used_data = used_data.select(range(n_sample_rows))
        else:
            used_data = self.original_data

        with log_time("map encode to excerpts"):
            processed_data = used_data.map(
                partial(self._encode),
                num_proc=self.dataloader_num_workers,
            )

        # TODO: add stratified splitting
        train_val_indices, test_indices = train_test_split(
            np.arange(len(processed_data)),
            test_size=0.1,
            shuffle=True,
            random_state=1234,
        )
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.1, random_state=1234
        )
        train = processed_data.select(train_indices).to_dict()  # list of <something>
        val = processed_data.select(val_indices).to_dict()
        test = processed_data.select(test_indices).to_dict()

        final_outputs = {
            "train": train,
            "val": val,
            "test": test,
            "tagname_to_id": self.tagname_to_tagid,
        }

        if save_split_dicts:
            dict_name = "sample_data.json" if testing else "full_data.json"

            with open(dict_name, "w") as fp:
                json.dump(final_outputs, fp)

        return final_outputs
