import logging
from ast import literal_eval
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from utils import revdict


class MultiHeadDataFrame(Dataset):
    """Creates a PyTorch dataset out of a Pandas DataFrame

    Args:
        dataframe: path to a DataFrame or directly a DataFrame
        tokenizer: tokenizer to pre-process source text
        source: textual source field that will be the input of models
        target: target classification field that will be the output of models
        groups: transforms target into a multi-target (each multi-label)
            that is, each sample is associated with 2D one-hot target matrix
        flatten: flatten group targets to 1D for convenience
    """

    def __init__(
        self,
        dataframe: Union[str, pd.DataFrame],
        tokenizer: PreTrainedTokenizer,
        source: str = "excerpt",
        target: str = "target",
        groups: Optional[List[List[str]]] = None,
        group_names: Optional[List[str]] = None,
        flatten: bool = True,
    ):
        self.logger = logging.getLogger()
        self.flatten = flatten

        # read dataframe manually if given as path
        if isinstance(dataframe, str):
            self.logger.info(f"Loading dataframe: {dataframe}")
            dataframe = pd.read_pickle(dataframe)

        # tokenize and save source data
        self.data = tokenizer(dataframe[source].tolist(), truncation=True, padding=True)

        # apply literal eval to have lists in target
        dataframe[target] = dataframe[target].apply(literal_eval)

        # prepare target encoding
        all_targets = np.hstack(dataframe[target].to_numpy())
        uniq_targets = np.unique(all_targets)

        # cluster into groups
        if groups:
            self.group_encoding = {t: idx for idx, group in enumerate(groups) for t in group}
            self.group_decoding = {idx: group for idx, group in enumerate(groups)}

            self.target_encoding = [{t: idx for idx, t in enumerate(group)} for group in groups]
            self.target_decoding = [revdict(encoding) for encoding in self.target_encoding]
            self.target_classes = [len(encoding.keys()) for encoding in self.target_encoding]
        else:
            self.group_encoding = {t: 0 for t in uniq_targets}
            self.group_decoding = {0: uniq_targets}

            self.target_encoding = {t: idx for idx, t in enumerate(uniq_targets)}
            self.target_encoding = revdict(self.target_encoding)
            self.target_classes = len(self.target_encoding.keys())

        self.logger.info(f"Automatically set target encodings: {self.target_encoding}")
        self.logger.info(f"Target size: [{self.target_classes}]")

        # prepare targets
        self.target = [self.onehot_encode(ts) for ts in dataframe[target].tolist()]

    def onehot_encode(self, targets: List[str]) -> Union[np.ndarray, List[np.ndarray]]:
        """Encodes given targets to one-hot representation"""

        onehot = [np.zeros(num_class, dtype=np.int) for num_class in self.target_classes]

        # 2D label matrix (group, class)
        for target in targets:
            group = self.group_encoding[target]
            encoding = self.target_encoding[group][target]
            onehot[group][encoding] = 1

        # flatten to 1D
        if self.flatten:
            onehot = np.concatenate(onehot)

        return onehot

    def onehot_decode(self, onehot: Union[np.ndarray, List[np.ndarray]]) -> List[str]:
        """Decodes given one-hot representation to targets"""

        # recover 2D label matrix
        if self.flatten:
            onehot = onehot.tolist()

            ind, _onehot = 0, []
            for num_class in self.target_classes:
                _onehot.append(onehot[ind : ind + num_class])
                ind += num_class
            onehot = _onehot

        # return positive labels
        return [
            self.target_decoding[i][j]
            for i in range(len(onehot))
            for j in range(onehot[i].shape[0])
            if onehot[i][j] == 1
        ]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.data.items()}

        if self.flatten:
            item["labels"] = torch.tensor(self.target[idx])
        else:
            item.update(
                {
                    f"labels_{i}": torch.tensor(self.target[idx][i])
                    for i in range(len(self.target_classes))
                }
            )

        return item
