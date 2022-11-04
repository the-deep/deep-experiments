import os

# setting tokenizers parallelism to false adds robustness when dploying the model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dill import needs to be kept for more robustness in multimodel serialization
import dill

dill.extend(True)

from typing import List
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


class CustomDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tagname_to_tagid,
        tokenizer,
        hypotheses: List[str],
        max_len: int = 128,
    ):
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
                self.entry_ids = list(dataframe["entry_id"])

        self.tagname_to_tagid = tagname_to_tagid
        self.tagid_to_tagname = list(tagname_to_tagid.keys())
        self.max_len = max_len
        self.hypotheses = hypotheses

    def encode_example(self, excerpt_text: str, index=None):

        ids = []
        mask = []
        token_type_ids = []

        for one_hypothesis in self.hypotheses:
            inputs = self.tokenizer(
                excerpt_text,
                one_hypothesis,
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                return_token_type_ids=True,
            )
            ids.append(torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(dim=0))
            mask.append(torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(dim=0))
            token_type_ids.append(
                torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(dim=0)
            )

        ids = torch.cat(ids, dim=0)
        mask = torch.cat(mask, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)

        encoded = {
            "ids": ids,
            "mask": mask,
            "token_type_ids": token_type_ids,
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
            encoded["entry_id"] = self.entry_ids[index]

        return encoded

    def __len__(self):
        return len(self.excerpt_text)

    def __getitem__(self, index):
        excerpt_text = str(self.excerpt_text[index])
        return self.encode_example(excerpt_text, index)