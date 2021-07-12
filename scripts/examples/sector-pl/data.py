import numpy as np

import torch
from torch.utils.data import Dataset


class SectorsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, class_to_id=None, max_len=200):
        self.tokenizer = tokenizer
        self.excerpt_text = list(dataframe["excerpt"])
        self.targets = list(dataframe["sectors"]) if "sectors" in dataframe.columns else None
        self.class_to_id = class_to_id
        self.max_len = max_len

    def encode_example(self, excerpt_text: str, index=None):
        # excerpt_text = " ".join(excerpt_text.split())

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

        if self.targets:
            target_indices = [
                self.class_to_id[target]
                for target in self.targets[index]
                if target in self.class_to_id
            ]
            targets = np.zeros(len(self.class_to_id), dtype=np.int)
            targets[target_indices] = 1

            encoded["targets"] = (
                torch.tensor(targets, dtype=torch.float32) if targets is not None else None
            )

        return encoded

    def __len__(self):
        return len(self.excerpt_text)

    def __getitem__(self, index):
        excerpt_text = str(self.excerpt_text[index])
        return self.encode_example(excerpt_text, index)
