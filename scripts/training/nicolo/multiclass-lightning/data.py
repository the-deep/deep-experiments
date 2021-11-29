import os

# setting tokenizers parallelism to false adds robustness when dploying the model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dill import needs to be kept for more robustness in multimodel serialization
import dill

dill.extend(True)

from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

from transformers import AutoModel


class CustomDataset(Dataset):
    def __init__(
        self, dataframe, tagname_to_tagid, tokenizer, model_name_or_path=None, max_len: int = 150
    ):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.automodel = AutoModel.from_pretrained(model_name_or_path)
        if dataframe is None:
            self.excerpt_text = None
        elif type(dataframe) is pd.Series:
            self.excerpt_text = dataframe.tolist()
        else:
            self.excerpt_text = dataframe["excerpt"].tolist()

        try:
            self.targets = list(dataframe["target"])
            self.entry_ids = list(dataframe["entry_id"])
        except Exception:
            self.targets = None
            self.entry_ids = None

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
        token_type_ids = inputs["token_type_ids"]

        encoded = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
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

            encoded["targets"] = torch.tensor(targets, dtype=float) if targets is not None else None
            encoded["entry_id"] = self.entry_ids[index]

        if as_batch:
            return {
                "ids": encoded["ids"].unsqueeze(0),
                "mask": encoded["mask"].unsqueeze(0),
                "token_type_ids": encoded["ids"].unsqueeze(0),
            }
        return encoded

    def __len__(self):
        return len(self.excerpt_text)

    def __getitem__(self, index):
        excerpt_text = str(self.excerpt_text[index])
        return self.encode_example(excerpt_text, index)
