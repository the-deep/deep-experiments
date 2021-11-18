from ast import literal_eval
import random
import numpy as np
import pandas as pd
from sklearn import metrics

import warnings

warnings.filterwarnings("ignore")

# GENERAL UTIL FUNCTIONS


def tagname_to_id(target):
    """
    Assign id to each tag
    """
    tag_set = set()
    for tags_i in target:
        tag_set.update(tags_i)
    tagname_to_tagid = {tag: i for i, tag in enumerate(list(sorted(tag_set)))}
    return tagname_to_tagid


def read_merge_data(TRAIN_PATH, TEST_PATH, data_format: str = "csv"):
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


def clean_rows(row):
    """
    1) Apply litteral evaluation
    2) keep unique values
    """
    return list(set(literal_eval(row)))


def flatten(t):
    return [item for sublist in t for item in sublist]


def custom_stratified_train_test_split(df, ratios):
    """
    custom function for stratified train test splitting
    1) take unique sub-tags (example: ['Health'])
    2) For each unique subtag:
        i) take all indexes that have that specific subtag
        ii) split them ransomly to train and test sets

    NB: a bit time consuming ~ 950 entries/second in average
    """
    train_ids = []
    val_ids = []
    positive_df = df.copy()

    unique_entries = list(np.unique(positive_df["target"].apply(str)))
    for entry in unique_entries:
        ids_entry = list(
            positive_df[positive_df.target.apply(str) == entry].entry_id.unique()
        )

        train_ids_entry = random.sample(
            ids_entry, int(len(ids_entry) * ratios["train"])
        )
        val_ids_entry = list(set(ids_entry) - set(train_ids_entry))

        train_ids.append(train_ids_entry)
        val_ids.append(val_ids_entry)

    return flatten(train_ids), flatten(val_ids)


def preprocess_df(
    df: pd.DataFrame,
    column_name: str,
    multiclass_bool: bool,
    keep_neg_labels: bool = False,
):

    """
    main preprocessing function:
    1) get positive entries using the porportions of train test split
    2) add negative exapmles using the ratios defined in the training notebook

    NB: work with ids because the augmented sentences have the same entry_id as the original ones
    """

    # rename column to 'target' to be able to work on it generically
    dataset = (
        df[["entry_id", "excerpt", column_name]]
        .rename(columns={column_name: "target"})
        .dropna()
        .copy()
    )

    dataset["target"] = dataset.target.apply(lambda x: clean_rows(x))

    if column_name == "sectors":
        dataset = dataset[dataset.target.apply(
            lambda x: "Cross" not in x
        )]
    if not multiclass_bool:
        dataset = dataset[dataset.target.apply(lambda x: len(x) == 1)]
    if not keep_neg_labels:
        dataset = dataset[dataset.target.apply(lambda x: len(x) > 0)]

    ratios = {
        "train": 0.9,
        "val": 0.1,
    }

    train_pos_entries, val_pos_entries = custom_stratified_train_test_split(
        dataset, ratios
    )

    df_train = dataset[dataset.entry_id.isin(train_pos_entries)]
    df_val = dataset[dataset.entry_id.isin(val_pos_entries)]

    return df_train, df_val


def stats_train_test(df_train: pd.DataFrame, df_val: pd.DataFrame, column_name: str):
    """
    Sanity check of data (proportion negative examples)
    """

    def compute_ratio_negative_positive(df):
        nb_rows_negative = df[df.target.apply(lambda x: len(x) == 0)].shape[0]
        if len(df) > 0:
            return np.round(nb_rows_negative / df.shape[0], 3)
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


def get_full_list_entries(df, tag):
    """
    Having a list of tags for each column,
    Return one list containing all tags in the column
    """
    pills_occurances = list()
    for pills in df[tag]:
        pills_occurances.extend(pills)
    return pills_occurances


def compute_weights(number_data_classes, n_tot):
    """
    weights computation for weighted loss function
    INPUTS:
    1) number_data_classes: list: number of samples for each class
    2) n_tot: total number of samples

    OUTPUT:
    list of weights used for training
    """
    number_classes = 2
    return list(
        [
            np.sqrt(n_tot / (number_classes * number_data_class))
            for number_data_class in number_data_classes
        ]
    )

def get_flat_labels (column_of_columns, tag_to_id, nb_subtags):
    matrix = [[
        1 if tag_to_id[i] in column else 0 for i in range (nb_subtags)
    ] for column in column_of_columns]
    return np.array(flatten(matrix))


    