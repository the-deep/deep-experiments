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
        group_names: name assoaciated with each classification head
        filter: (None, str, List of strings) filter dataset according
            to given group. `None` uses all of the data points. Single str key
            uses all data points with at least one target key value. If a list of
            strings is given, each key is used to check positivity of the sample,
            e.g., ['sector', 'pillar2d'] checks whether the data point has at
            least one target in `sector` or in `pillar2d` fields.
        flatten: flatten group targets to 1D for convenience
        online: online or offline tokenization
        inference: if True, does not process target or groups
        tokenizer_max_len: maximum output length for the tokenizer
    """

    def __init__(
        self,
        dataframe: Union[str, pd.DataFrame],
        tokenizer: PreTrainedTokenizer,
        source: str = "excerpt",
        target: str = "target",
        groups: Optional[List[List[str]]] = None,
        group_names: Optional[List[str]] = None,
        filter: Optional[Union[str, List[str]]] = None,
        flatten: bool = True,
        online: bool = False,
        inference: bool = False,
        tokenizer_max_len: int = 200,
    ):
        self.group_names = group_names
        self.flatten = flatten
        self.tokenizer = tokenizer
        self.online = online
        self.inference = inference
        self.logger = logging.getLogger()

        # read dataframe manually if given as path
        if isinstance(dataframe, str):
            self.logger.info(f"Loading dataframe: {dataframe}")
            dataframe = pd.read_pickle(dataframe)

        # cast filter to array
        if isinstance(filter, str):
            filter = [filter]

        # filter data frame
        if filter is not None:
            pos = np.zeros(len(dataframe), dtype=np.bool)
            for f in filter:
                pos |= np.array([len(item) > 0 for item in dataframe[f].tolist()], dtype=np.bool)
            dataframe = dataframe[pos]
            self.logger.info(f"Filtered data points with non-empty (or) {','.join(filter)} values")

        # prepare tokenizer options
        self.tokenizer_options = {
            "truncation": True,
            "padding": "max_length",
            "add_special_tokens": True,
            "return_token_type_ids": True,
            "max_length": min(tokenizer_max_len, tokenizer.model_max_length),
        }
        if tokenizer.model_max_length < tokenizer_max_len:
            self.logger.info(
                f"Using maximum model length: {tokenizer.model_max_length} instead"
                f"of given length: {tokenizer_max_len}"
            )

        if self.online:
            # ensure that we are in training
            assert not self.inference, "Online tokenization is only supported in training-time"

            # save data as exceprt
            self.data = dataframe[source].tolist()
        else:
            # tokenize and save source data
            self.logger.info("Applying offline tokenization")
            self.data = tokenizer(dataframe[source].tolist(), **self.tokenizer_options)

        if not self.inference:
            # apply literal eval to have lists in target
            dataframe[target] = dataframe[target].apply(literal_eval)

            # prepare target encoding
            all_targets = np.hstack(dataframe[target].to_numpy())
            uniq_targets = np.unique(all_targets)

            if groups:
                # process given groups
                self.group_encoding = {t: idx for idx, group in enumerate(groups) for t in group}
                self.group_decoding = {idx: group for idx, group in enumerate(groups)}

                self.target_encoding = [{t: idx for idx, t in enumerate(group)} for group in groups]
                self.target_decoding = [revdict(encoding) for encoding in self.target_encoding]
                self.target_classes = [len(encoding.keys()) for encoding in self.target_encoding]
            else:
                # single group encoding - decoding
                self.group_encoding = {t: 0 for t in uniq_targets}
                self.group_decoding = {0: uniq_targets}

                self.target_encoding = {t: idx for idx, t in enumerate(uniq_targets)}
                self.target_encoding = revdict(self.target_encoding)
                self.target_classes = [len(self.target_encoding.keys())]

                self.logger.info(f"Using target encodings: {self.target_encoding}")
                self.logger.info(f"Target size: [{self.target_classes}]")

            # prepare targets
            self.target = [self.onehot_encode(ts) for ts in dataframe[target].tolist()]

            # prepare group targets
            if groups:
                self.group = [self.group_encode(ts) for ts in dataframe[target].tolist()]

    def group_encode(self, targets: List[str]) -> np.ndarray:
        """Encodes given targets to group representation"""

        onehot = np.zeros(len(self.target_classes), dtype=np.bool)

        # flip all groups
        for target in targets:
            onehot[self.group_encoding[target]] = 1

        return onehot

    def onehot_encode(self, targets: List[str]) -> Union[np.ndarray, List[np.ndarray]]:
        """Encodes given targets to one-hot representation"""

        onehot = [np.zeros(num_class, dtype=np.bool) for num_class in self.target_classes]

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
        if self.online:
            data = self.tokenizer(self.data[idx : idx + 1], **self.tokenizer_options)
            item = {key: torch.tensor(val[0]) for key, val in data.items()}
        else:
            item = {key: torch.tensor(val[idx]) for key, val in self.data.items()}

        if self.inference:
            if self.flatten:
                item["labels"] = torch.tensor(self.target[idx])
            else:
                item.update(
                    {
                        f"labels_{self.group_names[i]}": torch.tensor(self.target[idx][i])
                        for i in range(len(self.target_classes))
                    }
                )

            if self.group is not None:
                item["groups"] = torch.tensor(self.group[idx])

        return item
