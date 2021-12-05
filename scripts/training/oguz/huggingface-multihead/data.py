import logging
from ast import literal_eval
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from utils import revdict, read_dataframe


class TextDataFrame(Dataset):
    """Creates a PyTorch dataset out of a text field inside a Pandas DataFrame.

    Args:
        dataframe: path to a DataFrame or a DataFrame
        tokenizer: tokenizer to pre-process source text
        source: textual source field that will be the input of models
        online: online or offline tokenization
        tokenizer_max_len: maximum output length for the tokenizer
    """

    def __init__(
        self,
        dataframe: Union[str, pd.DataFrame],
        tokenizer: PreTrainedTokenizer,
        source: str = "excerpt",
        online: bool = False,
        tokenizer_max_len: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.online = online
        self.logger = logging.getLogger()

        # read dataframe manually if given as path
        if isinstance(dataframe, str):
            self.logger.info(f"Loading dataframe: {dataframe}")
            dataframe = read_dataframe(dataframe)

        # prepare tokenizer options
        self.tokenizer_options = {
            "truncation": True,
            "padding": True,
        }
        if tokenizer_max_len:
            self.tokenizer_options.update(
                {
                    "padding": "max_length",
                    "max_length": min(tokenizer_max_len, tokenizer.model_max_length),
                }
            )
            if tokenizer.model_max_length < tokenizer_max_len:
                self.logger.info(
                    f"Using maximum model length: {tokenizer.model_max_length} instead"
                    f"of given length: {tokenizer_max_len}"
                )

        # save data as exceprt
        self.data = dataframe[source].tolist()
        self.data_len = len(self.data)

        if not self.online:
            # tokenize and save source data
            self.logger.info("Applying offline tokenization")
            self.data = tokenizer(self.data, **self.tokenizer_options)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.online:
            data = self.tokenizer(self.data[idx : idx + 1], **self.tokenizer_options)
            item = {key: torch.tensor(val[0]) for key, val in data.items()}
        else:
            item = {key: torch.tensor(val[idx]) for key, val in self.data.items()}

        return item


class MultiTargetDataFrame(Dataset):
    """Creates a PyTorch dataset out of a field containing list of labels
    for a multi-label classification problem out of a Pandas DataFrame.


    Args:
        dataframe: path to a DataFrame or a DataFrame
        target: target classification field that will be the output of models
        groups: transforms target into a multi-target problem (each multi-label)
            that is, each sample is associated with 2D one-hot target matrix
            e.g., 6 label classification with two groups: [A, B, C], [D, E, F]
        group_names: name assoaciated with each classification head
            e.g., 2 group names: ABC and DEF
        exclude: omit the given target labels.
        flatten: flatten targets to 1D for convenience
    """

    def __init__(
        self,
        dataframe: Union[str, pd.DataFrame],
        target: str = "target",
        groups: Optional[List[List[str]]] = None,
        group_names: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        flatten: bool = True,
    ):
        # process groups
        if groups is not None:
            if group_names is not None:
                assert len(groups) == len(
                    group_names
                ), "Group names should be at equal length with groups"
            else:
                group_names = [f"Group {i}" for i in range(len(groups))]
        else:
            assert flatten, "Use flatten if no group information is provided"

        self.groups = groups
        self.group_names = group_names
        self.flatten = flatten
        self.logger = logging.getLogger()

        # read dataframe manually if given as path
        if isinstance(dataframe, str):
            self.logger.info(f"Loading dataframe: {dataframe}")
            dataframe = read_dataframe(dataframe)

        # apply literal eval to have lists in target
        if not isinstance(dataframe[target].iloc[0], list):
            dataframe[target] = dataframe[target].apply(literal_eval)

        # omit the given exclude labels
        if exclude:
            dataframe[target] = [
                [label for label in labels if label not in exclude]
                for labels in dataframe[target].tolist()
            ]

        if groups:
            # process given groups
            self.group_encoding = {t: idx for idx, group in enumerate(groups) for t in group}
            self.group_decoding = {idx: group for idx, group in enumerate(groups)}

            self.target_encoding = [{t: idx for idx, t in enumerate(group)} for group in groups]
            self.target_decoding = [revdict(encoding) for encoding in self.target_encoding]
            self.target_classes = [len(encoding.keys()) for encoding in self.target_encoding]
        else:
            # prepare target encoding
            all_targets = np.hstack(dataframe[target].to_numpy())
            uniq_targets = np.unique(all_targets)

            # single group encoding - decoding
            self.group_encoding = {t: 0 for t in uniq_targets}
            self.group_decoding = {0: uniq_targets}
            self.groups = [uniq_targets.tolist()]
            self.group_names = ["ALL"]

            self.target_encoding = {t: idx for idx, t in enumerate(uniq_targets)}
            self.target_encoding = revdict(self.target_encoding)
            self.target_classes = [len(self.target_encoding.keys())]

            self.logger.info(f"Using target encodings: {self.target_encoding}")
            self.logger.info(f"Target size: [{self.target_classes}]")

        # prepare targets
        self.target = [self.onehot_encode(ts) for ts in dataframe[target].tolist()]
        self.data_len = len(self.target)

        # prepare group targets
        if groups:
            self.group = [self.group_encode(ts) for ts in dataframe[target].tolist()]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        item = {}

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

    def compute_stats(self) -> Dict[str, int]:
        """Computes occurences of each target and group"""

        counts = {}
        classes = [target for group in self.groups for target in group]
        if self.flatten:
            sums = np.sum(np.stack(self.target, axis=-1), axis=-1)
            counts.update({c: s for c, s in zip(classes, sums.tolist())})
        else:
            for i, _ in enumerate(self.group_names):
                targets = [target[i] for target in self.target]
                sums = np.sum(np.stack(targets, axis=-1), axis=-1)
                counts.update({c: s for c, s in zip(self.groups[i], sums)})

        for i, group_name in enumerate(self.group_names):
            counts.update(
                {group_name: np.sum(np.array([counts[group] for group in self.groups[i]]))}
            )
        counts.update(
            {"ALL": np.sum(np.array([counts[group_name] for group_name in self.group_names]))}
        )
        return counts

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


class MultiHeadDataFrame(Dataset):
    """Creates a PyTorch dataset out of a Pandas DataFrame that supports
    multi-head classification tasks where each fine-grained label belongs
    to a super-category or a group.


    Args:
        dataframe: path to a DataFrame or a DataFrame
        tokenizer: tokenizer to pre-process source text
        source: textual source field that will be the input of models
        targets: target classification fields that will be the output of models
        groups: transforms target into a multi-target (each multi-label)
            that is, each sample is associated with 2D one-hot target matrix
        group_names: name assoaciated with each classification head
        filter: (None, str, List of strings) filter dataset according
            to given group. `None` uses all of the data points. Single str key
            uses all data points with at least one target key value. If a list of
            strings is given, each key is used to check positivity of the sample,
            e.g., ['sector', 'pillar2d'] checks whether the data point has at
            least one target in `sector` or in `pillar2d` fields.
        exclude: (None, List of strings, List of List of strings) omit the given
            targets. For multi-target classification, expects a list with
            elements of lists.
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
        targets: Union[str, List[str]] = "target",
        groups: Optional[Union[List[List[str]], List[List[List[str]]]]] = None,
        group_names: Optional[Union[List[str], List[List[str]]]] = None,
        exclude: Optional[List[str]] = None,
        filter: Optional[Union[str, List[str]]] = None,
        flatten: bool = True,
        online: bool = False,
        inference: bool = False,
        tokenizer_max_len: Optional[int] = None,
    ):
        self.logger = logging.getLogger()

        if online:
            # ensure that we are in training
            assert not inference, "Online tokenization is only supported in training-time"

        # read dataframe manually if given as path
        if isinstance(dataframe, str):
            self.logger.info(f"Loading dataframe: {dataframe}")
            dataframe = read_dataframe(dataframe)

        # cast filter to array
        if isinstance(filter, str):
            filter = [filter]

        # filter data frame
        if filter is not None:
            pos = np.zeros(len(dataframe), dtype=np.bool)
            for f in filter:
                # apply literal eval to have lists
                dataframe[f] = dataframe[f].apply(literal_eval)

                # get positive fields
                pos |= np.array([len(item) > 0 for item in dataframe[f].tolist()], dtype=np.bool)

            # filter negative rows
            dataframe = dataframe[pos]
            self.logger.info(
                f"Filtered data points with non-empty {','.join(filter)} values"
                "(using 'or' if multiple fields)"
            )

        # prepare text source data
        self.data = TextDataFrame(
            dataframe=dataframe,
            tokenizer=tokenizer,
            source=source,
            online=online,
            tokenizer_max_len=tokenizer_max_len,
        )

        # prepare targets
        if isinstance(targets, str):
            # assert isinstance(groups, List[List[str]]), "Expecting `groups` to be a list of lists"
            # assert isinstance(group_names, List[str]), "Expecting `group_names` to be a list"

            targets = [targets]
            groups = [groups]
            group_names = [group_names]
            self.single = True
        else:
            # assert isinstance(
            #    groups, List[List[List[str]]]
            # ), "Expecting `groups` to be a list of lists of lists"
            # assert isinstance(
            #    group_names, List[List[str]]
            # ), "Expecting `group_names` to be a list of lists"
            self.single = False

        # prepare omit lists
        if exclude is None:
            exclude = [None for target in targets]

        self.targets = []
        if not inference:
            for _target, _groups, _group_names, _exclude in zip(
                targets, groups, group_names, exclude
            ):
                self.targets.append(
                    MultiTargetDataFrame(
                        dataframe=dataframe,
                        target=_target,
                        groups=_groups,
                        group_names=_group_names,
                        exclude=_exclude,
                        flatten=flatten,
                    )
                )
                assert len(self.data) == len(
                    self.targets[-1]
                ), "Text source and target have different lengths!"
        self.data_len = self.data.data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.single:
            item.update(self.target[0])
        else:
            for i, target in enumerate(self.targets):
                item.update({(f"head{i}_" + k): v for k, v in target[idx].items()})

        return item
