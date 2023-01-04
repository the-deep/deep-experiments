from ast import literal_eval
import random
from string import punctuation
import numpy as np
import pandas as pd
import re
from copy import copy
import warnings
from typing import Dict, List, Union, Tuple
from collections import Counter
import torch
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# GENERAL UTIL FUNCTIONS


def map_id_layer_to_level(ids_each_level) -> Dict[int, int]:
    dict_layers = {}
    lengthes = [len(id_one_level) for id_one_level in ids_each_level]
    tag_id = 0
    for i, length_tmp in enumerate(lengthes):
        for j in range(length_tmp):
            dict_layers[tag_id] = i
            tag_id += 1
    return dict_layers


def beta_score(precision: float, recall: float, f_beta: Union[int, float]) -> float:
    """get beta score from precision and recall"""
    return (1 + f_beta**2) * precision * recall / ((f_beta**2) * precision + recall)


def clean_name_for_logging(
    dict_values: Dict[str, float], context: str, af_id: int = None
) -> Dict[str, float]:
    """clean names and prepare them for logging"""

    def get_new_name(name: str, context: str, af_id: int = None):
        # clean regex
        claned_name = re.sub("[^0-9a-zA-Z]+", "_", name)
        if af_id is None:
            return f"{context}_{claned_name}"
        else:
            return f"{context}_{claned_name}_{af_id}"

    return {
        get_new_name(key, context, af_id): value for key, value in dict_values.items()
    }


def get_tagname_to_id(target) -> Dict[str, int]:
    """
    Assign id to each tag
    """
    tag_set = set()
    for tags_i in target:
        tag_set.update(tags_i)
    tagname_to_tagid = {tag: i for i, tag in enumerate(list(sorted(tag_set)))}
    return tagname_to_tagid


def custom_eval(x) -> List:
    if str(x) == "nan":
        return []
    if str(x) == "[None]":
        return []
    if type(x) == list:
        return x
    else:
        return literal_eval(x)


def read_merge_data(
    TRAIN_PATH: str, TEST_PATH: str, data_format: str = "csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    read data as csv or pickle, then merge it
    """

    if data_format == "pickle":
        train_df = pd.read_pickle(f"{TRAIN_PATH}/train.pickle")
        test_df = pd.read_pickle(f"{TEST_PATH}/val.pickle")

    else:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def clean_rows(row: List[str]) -> List[str]:
    """
    1) Apply litteral evaluation
    2) keep unique values
    """
    return list(set(literal_eval(row)))


def flatten(t: List[List]) -> List:
    """flatten list of lists"""
    return [item for sublist in t for item in sublist]


def custom_stratified_train_test_split(
    df: pd.DataFrame, ratios: float
) -> Tuple[List[int], List[int]]:
    """
    custom function for stratified train test splitting
    1) take unique sub-tags (example: ['Health'])
    2) For each unique subtag:
        i) take all indexes that have that specific subtag
        ii) split them randomly to train and test sets
    """
    train_ids = []
    val_ids = []
    positive_df = df.copy()
    positive_df["target"] = positive_df["target"].apply(str)
    ids = positive_df.groupby("target")["entry_id"].agg(list).values
    unique_ids = [list(set(list_)) for list_ in ids]

    for ids_entry in unique_ids:

        train_ids_entry = random.sample(
            ids_entry, int(len(ids_entry) * ratios["train"])
        )
        val_ids_entry = list(set(ids_entry) - set(train_ids_entry))

        train_ids.append(train_ids_entry)
        val_ids.append(val_ids_entry)

    return flatten(train_ids), flatten(val_ids)


def preprocess_df(
    df: pd.DataFrame, relabeled_columns: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    main preprocessing function:
    1) get positive entries using the porportions of train test split
    2) add negative exapmles using the ratios defined in the training notebook

    NB: work with ids because the augmented sentences have the same entry_id as the original ones
    """
    # rename column to 'target' to be able to work on it generically
    dataset = df.dropna().copy()

    bool_augmented_data = dataset.entry_id == 0

    dataset["target"] = dataset.target.apply(lambda x: clean_rows(x))

    dataset["target"] = dataset.target.apply(
        lambda x: [item for item in x if item != "NOT_MAPPED"]
    )

    # For relabeling data
    if relabeled_columns != "none":

        if relabeled_columns == "sectors":
            dataset["target"] = dataset.target.apply(
                lambda x: [
                    item for item in x if "sectors" in item and "Cross" not in item
                ]
            )

        if relabeled_columns == "secondary_tags":
            dataset["target"] = dataset.target.apply(
                lambda x: [
                    item
                    for item in x
                    if "secondary_tags" in item and "severity" not in item
                ]
            )

        # TODO: subsectors

    ratios = {
        "train": 0.85,
        "val": 0.15,
    }

    train_pos_entries, val_pos_entries = custom_stratified_train_test_split(
        dataset[~bool_augmented_data], ratios
    )

    df_train = dataset[dataset.entry_id.isin(train_pos_entries)]
    # df_train = pd.concat([df_train, dataset[bool_augmented_data]])
    df_val = dataset[dataset.entry_id.isin(val_pos_entries)]

    return df_train, df_val


def stats_train_test(
    df_train: pd.DataFrame, df_val: pd.DataFrame, column_name: str
) -> float:
    """
    Sanity check of data (proportion negative examples)
    """

    def compute_ratio_negative_positive(df):
        nb_rows_negative = df[df.target.apply(lambda x: len(x) == 0)].shape[0]
        if len(df) > 0:
            return np.round(nb_rows_negative / df.shape[0], 2)
        else:
            return 0

    ratio_negative_positive = {
        f"ratio_negative_examples_train_{column_name}": compute_ratio_negative_positive(
            df_train
        ),
        f"ratio_negative_examples_val_{column_name}": compute_ratio_negative_positive(
            df_val
        ),
    }

    return ratio_negative_positive


def get_tags_proportions(
    tagname_to_tagid: Dict[str, int], targets_list: List[str]
) -> torch.Tensor:
    """get alphas for BCE weighted loss"""
    counts = dict(Counter(flatten(targets_list)))
    sorted_counts = [counts[k] for k, v in tagname_to_tagid.items()]
    return torch.tensor(
        compute_weights(number_data_classes=sorted_counts, n_tot=len(targets_list)),
        dtype=torch.float64,
    )


def compute_weights(number_data_classes: List[int], n_tot: int) -> List[float]:
    """
    weights computation for weighted loss function
    INPUTS:
    1) number_data_classes: list: number of samples for each class
    2) n_tot: total number of samples

    OUTPUT:
    list of weights used for training
    """
    return [number_data_class / n_tot for number_data_class in number_data_classes]


def get_flat_labels(column_of_columns, tag_to_id: Dict[str, int], nb_subtags: int):
    matrix = [
        [1 if tag_to_id[i] in column else 0 for i in range(nb_subtags)]
        for column in column_of_columns
    ]
    return np.array(flatten(matrix))


def get_tag_id_to_layer_id(ids_each_level):
    tag_id = 0
    list_id = 0
    tag_to_list = {}
    for id_list in ids_each_level:
        for i in range(len(id_list)):
            tag_to_list.update({tag_id + i: list_id})
        tag_id += len(id_list)
        list_id += 1
    return tag_to_list


def get_first_level_ids(tagname_to_tagid: Dict[str, int]) -> List[List[List[int]]]:
    """having list of unique labels, create the labels ids in different lists"""
    all_names = list(tagname_to_tagid.keys())
    split_names = [name.split("->") for name in all_names]

    assert np.all([len(name_list) == 3 for name_list in split_names])
    final_ids = []

    tag_id = 0
    first_level_names = list(np.unique([name_list[0] for name_list in split_names]))
    for first_level_name in first_level_names:
        first_level_ids = []
        kept_names = [
            name_list[1:]
            for name_list in split_names
            if name_list[0] == first_level_name
        ]
        second_level_names = list(np.unique([name[0] for name in kept_names]))
        for second_level_name in second_level_names:
            second_level_ids = []
            third_level_names = [
                name_list[1]
                for name_list in kept_names
                if name_list[0] == second_level_name
            ]
            for _ in range(len(third_level_names)):
                second_level_ids.append(tag_id)
                tag_id += 1
            first_level_ids.append(second_level_ids)
        final_ids.append(first_level_ids)

    return final_ids


def get_relevant_labels(target_column, min_kept_ratio: float = 0.02) -> List[str]:
    n_items = len(target_column)
    targets = flatten(
        target_column.apply(
            lambda x: [item for item in custom_eval(x) if item != "NOT_MAPPED"]
        ).tolist()
    )
    relevant_labels = [
        label_name
        for label_name, label_counts in dict(Counter(targets)).items()
        if (label_counts / n_items) > min_kept_ratio
    ]
    return relevant_labels


def delete_punctuation(text: str) -> str:
    clean_text = copy(text)
    to_be_cleaned_punctuations = [
        # ",",
        # ";",
        "[",
        "]",
        "(",
        ")",
        "-",
        "_",
        "{",
        "}",
        "+",
        "*",
        "=",
        "<",
        ">",
        "/",
        # "!",
        # "?",
    ]
    for punct in to_be_cleaned_punctuations:
        clean_text = clean_text.replace(punct, "").replace("...", ".")
    return clean_text.rstrip().lstrip()


def preprocess_one_sentence(sentence):
    """
    function to preprocess one_sentence:
        - lower and remove punctuation
    """

    def RepresentsInt(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    if type(sentence) is not str:
        sentence = str(sentence)

    new_words = []
    words = sentence.split()

    for word in words:
        # keep clean words and remove hyperlinks
        bool_word_not_empty_string = word != ""

        # TODO: treat better stop words for no bias
        # bool_word_not_stop_word = word.lower() not in stop_words

        if bool_word_not_empty_string:  # and bool_word_not_stop_word:

            # TODO: better number preprocessing
            # if word.isdigit():
            #    appended_word = "^"

            # TODO: location preprocessing
            # elif word_is_location(word):
            #    appended_word = "`"

            appended_word = word

            new_words.append(appended_word)

    return " ".join(new_words).rstrip().lstrip()


def get_n_tokens(
    text: List[str], tokenizer_name: str, batch_size_tokenizer: int = 128
) -> np.ndarray:
    """
    get number of tokens after tokeniziation for excerpts.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    masks = []
    for i in range(0, len(text), batch_size_tokenizer):

        one_batch = text[i : i + batch_size_tokenizer]

        masks.append(
            tokenizer(
                one_batch,
                None,
                truncation=False,
                add_special_tokens=True,
                padding="max_length",
                return_token_type_ids=False,
            )["attention_mask"]
        )

    lengths = np.array(
        [
            np.sum(np.array(one_mask) == 1)
            for masks_sublist in masks
            for one_mask in masks_sublist
        ]
    )

    return lengths
