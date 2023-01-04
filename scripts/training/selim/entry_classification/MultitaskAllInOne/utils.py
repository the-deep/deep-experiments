from ast import literal_eval
import random
import numpy as np
import pandas as pd
import re
import warnings

warnings.filterwarnings("ignore")

# GENERAL UTIL FUNCTIONS


def map_id_layer_to_level(ids_each_level):
    dict_layers = {}
    lengthes = [len(id_one_level) for id_one_level in ids_each_level]
    tag_id = 0
    for i, length_tmp in enumerate(lengthes):
        for j in range(length_tmp):
            dict_layers[tag_id] = i
            tag_id += 1
    return dict_layers


def beta_score(precision, recall, f_beta):
    return (1 + f_beta ** 2) * precision * recall / ((f_beta ** 2) * precision + recall)


def get_new_name(name, context):
    # clean regex
    claned_name = re.sub("[^0-9a-zA-Z]+", "_", name)
    return f"{context}_{claned_name}"


def clean_name_for_logging(dict_values, context):
    return {get_new_name(key, context): value for key, value in dict_values.items()}


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
        ii) split them randomly to train and test sets
    """
    train_ids = []
    val_ids = []
    positive_df = df.copy()
    positive_df["target"] = positive_df["target"].apply(str)
    ids = positive_df.groupby("target")["entry_id"].agg(list).values
    unique_ids = [list(np.unique(list_)) for list_ in ids]

    for ids_entry in unique_ids:

        train_ids_entry = random.sample(
            ids_entry, int(len(ids_entry) * ratios["train"])
        )
        val_ids_entry = list(set(ids_entry) - set(train_ids_entry))

        train_ids.append(train_ids_entry)
        val_ids.append(val_ids_entry)

    return flatten(train_ids), flatten(val_ids)


def preprocess_df(df: pd.DataFrame, relabeled_columns: str):

    """
    main preprocessing function:
    1) get positive entries using the porportions of train test split
    2) add negative exapmles using the ratios defined in the training notebook

    NB: work with ids because the augmented sentences have the same entry_id as the original ones
    """
    # rename column to 'target' to be able to work on it generically
    dataset = df[["entry_id", "excerpt", "target"]].dropna().copy()

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
    return [
        1 - (number_data_class / n_tot) for number_data_class in number_data_classes
    ]


def get_flat_labels(column_of_columns, tag_to_id, nb_subtags):
    matrix = [
        [1 if tag_to_id[i] in column else 0 for i in range(nb_subtags)]
        for column in column_of_columns
    ]
    return np.array(flatten(matrix))


def get_preds_entry(preds_column, return_at_least_one=True, ratio_nb=1):
    preds_entry = [
        sub_tag
        for sub_tag in list(preds_column.keys())
        if preds_column[sub_tag] > ratio_nb
    ]
    if return_at_least_one:
        if len(preds_entry) == 0:
            preds_entry = [
                sub_tag
                for sub_tag in list(preds_column.keys())
                if preds_column[sub_tag] == max(list(preds_column.values()))
            ]
    return preds_entry

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