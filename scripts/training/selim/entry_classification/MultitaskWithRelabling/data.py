from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from copy import copy
from string import punctuation
import torch


def _delete_context(text: str) -> str:
    if text[0] == "[":
        n_strings = len(text)
        hook_end = 0
        while text[hook_end] != "]" and hook_end < 50:
            hook_end += 1
            if hook_end == (n_strings - 1):
                return text

        if hook_end < 70:
            return text[(hook_end + 1) :]
        else:
            return text

    else:
        return text


def _clean_lgbt_words(text: str) -> str:
    def lgbt_in_word(word):
        return "lgbt" in word.lower() or "lgtb" in word.lower()

    def get_one_lgbt_token(word):
        if word[-1] in punctuation and word[-1] != "+":
            final_punct = word[-1]
            word = word[:-1]
        else:
            final_punct = ""

        return ("lgbt" + final_punct).rstrip()

    clean_text = copy(text)
    if lgbt_in_word(clean_text):
        clean_text = clean_text.replace(" +", "+")
        words = clean_text.split(" ")
        output_text = " ".join(
            [
                one_word if not lgbt_in_word(one_word) else get_one_lgbt_token(one_word)
                for one_word in words
            ]
        )
        return output_text

    else:
        return text


def _preprocess_text(text: str) -> str:
    clean_text = copy(text).strip()
    # clean_text = _delete_context(clean_text)
    # clean_text = delete_punctuation(clean_text)
    clean_text = _clean_lgbt_words(clean_text)
    # clean_text = preprocess_one_sentence(clean_text)
    return clean_text


class ExcerptsDataset(Dataset):
    """
    transformers custom dataset
    """

    def __init__(self, dataframe, tagname_to_tagid, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.data = dataframe

        self.targets = None
        self.entry_ids = None

        if dataframe is None:
            self.excerpt_text = None

        elif type(dataframe) is str:
            self.excerpt_text = [dataframe]

        elif type(dataframe) is list:
            self.excerpt_text = dataframe

        elif type(dataframe) is pd.Series:
            self.excerpt_text = dataframe.tolist()

        else:
            self.excerpt_text = dataframe["excerpt"].tolist()
            df_cols = dataframe.columns
            if "target" in df_cols and "entry_id" in df_cols:
                self.targets = list(dataframe["target"])
                # self.entry_ids = list(dataframe["entry_id"])

        self.tagname_to_tagid = tagname_to_tagid
        self.tagid_to_tagname = list(tagname_to_tagid.keys())
        self.max_len = max_len

    def encode_example(self, excerpt_text: str, index=None, as_batch: bool = False):

        inputs = self.tokenizer(
            excerpt_text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        encoded = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
        }

        targets = None
        if self.targets:
            target_indices = [
                self.tagname_to_tagid[target]
                for target in self.targets[index]
                if target in self.tagname_to_tagid
            ]
            targets = np.zeros(len(self.tagname_to_tagid), dtype=int)
            targets[target_indices] = 1

            encoded["targets"] = (
                torch.tensor(targets, dtype=float) if targets is not None else None
            )

        return encoded

    def __len__(self):
        return len(self.excerpt_text)

    def __getitem__(self, index):
        excerpt_text = str(self.excerpt_text[index])
        cleaned_excerpt = _preprocess_text(excerpt_text)
        return self.encode_example(cleaned_excerpt, index)
