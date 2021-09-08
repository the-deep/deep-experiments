from ast import literal_eval

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from skmultilearn.model_selection import iterative_train_test_split

import warnings

warnings.filterwarnings("ignore")

# GENERAL UTIL FUNCTIONS


def clean_rows(row):
    """
    1) Apply litteral evaluation
    2) Drop values that are repeated multiple times in rows
    """
    return list(set(literal_eval(row)))
    
def tagname_to_id(target):
    """
    Assign id to each tag
    """
    tag_set = set()
    for tags_i in target:
        tag_set.update(tags_i)
    tagname_to_tagid = {tag: i for i, tag in enumerate(list(sorted(tag_set)))}
    return tagname_to_tagid


def read_merge_data(
    TRAIN_PATH, VAL_PATH, data_format: str = "csv"
):

    if data_format == "pickle":
        train_df = pd.read_pickle(f"{TRAIN_PATH}/train.pickle")
        val_df = pd.read_pickle(f"{VAL_PATH}/val.pickle")

    else:
        train_df = pd.read_csv(TRAIN_PATH)
        val_df = pd.read_csv(VAL_PATH)

    all_dataset = pd.concat([train_df, val_df])

    return all_dataset

def preprocess_df(df:pd.DataFrame, column_name:str):
    all_dataset = df.copy()
    # Keep only unique values in pillars
    all_dataset[column_name] = all_dataset[column_name].apply(lambda x: clean_rows(x))

    # Keep only rows with a not empty pillar
    all_dataset = all_dataset[all_dataset[column_name].apply(lambda x: len(x) > 0)][
        ["entry_id", "excerpt", column_name]
    ].rename(columns={column_name: "target"}).dropna()

    return custom_train_test_df_split(all_dataset)

# DATA PREPROCESSING AND AUGMENTATION
def get_occurences (df, tag):
    pills_occurances = list()
    for pills in df[tag]:
        pills_occurances.extend(pills)
    return pills_occurances

def custom_train_test_df_split(df:pd.DataFrame, tag:str='target'):
    classes = list(np.unique(get_occurences (df, tag)))
    class_to_id = {clss: i for i, clss in enumerate(classes)}
    num_classes = len(classes)
    labels = np.zeros([len(df), num_classes])
    for i, sectors_i in enumerate (df[tag]):
        for sec in sectors_i:
            labels[i, class_to_id[sec]] = 1

    ##splitting
    X_train, y_train, X_test, y_test = iterative_train_test_split(
        df["entry_id"].to_numpy().reshape(-1, 1), labels, test_size=0.3)

    bool_isin_ids = df.entry_id.isin(X_train.flatten())
    df_train = df[bool_isin_ids]
    df_test = df[~bool_isin_ids]

    return df_train, df_test





"""def preprocess_data(
    dataset,
    n_synonym_augmenter=1,
    n_swap=1,
    perform_augmentation: bool = True,
    method="keep all",
    language_chosen: str = "en",
):
    
    #1) filter with respect to language
    #2) perform augmentation
    #3) split to training and test set
    

    df = dataset.copy()

    if method == "keep en":
        df = df[df.language == language_chosen]
    elif method == "omit en":
        df = df[df.language != language_chosen]

    df = df[["entry_id", "excerpt", "target"]]

    if perform_augmentation:
        train_data, test_data = train_test_split(df, test_size=0.3)
        return augment_data(train_data, n_synonym_augmenter, n_swap), test_data
    else:
        return train_test_split(df, test_size=0.2)"""


"""def augment_data(df, n_synonym, n_swap):
    # 1) Augment with synonym
    # 2) Apply swap on new (augmented with synonym) dataframe
    if n_synonym:
        syn_aug_en = naw.SynonymAug(lang="eng", aug_min=3, aug_p=0.4)
        syn_aug_fr = naw.SynonymAug(lang="fra", aug_min=3, aug_p=0.4)
        syn_aug_es = naw.SynonymAug(lang="spa", aug_min=3, aug_p=0.4)

        en_syn = df[df.language == "en"]
        fr_syn = df[df.language == "fr"]
        es_syn = df[df.language == "es"]

        en_syn.excerpt = en_syn.excerpt.apply(lambda x: syn_aug_en.augment(x, n=n_synonym))
        fr_syn.excerpt = fr_syn.excerpt.apply(lambda x: syn_aug_fr.augment(x, n=n_synonym))
        es_syn.excerpt = es_syn.excerpt.apply(lambda x: syn_aug_es.augment(x, n=n_synonym))

        whole_synoynm = pd.concat([en_syn, fr_syn, es_syn])

        for _, row in whole_synoynm.iterrows():
            excerpts = row.excerpt
            for i in range(0, n_synonym):
                row.excerpt = excerpts[i]
                df = df.append(row)

    if n_swap:
        swap = naw.RandomWordAug(action="swap", aug_min=3, aug_max=5)
        swap_df = df
        swap_df.excerpt = swap_df.excerpt.apply(lambda x: swap.augment(x, n=n_swap))

        for _, row in swap_df.iterrows():
            excerpts = row.excerpt
            for i in range(0, n_swap):
                row.excerpt = excerpts[i]
                df = df.append(row)

    return df
"""

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


# EVALUATION


def perfectEval(anonstring):
    try:
        ev = literal_eval(anonstring)
        return ev
    except ValueError:
        corrected = "'" + anonstring + "'"
        ev = literal_eval(corrected)
        return ev


def fill_column(row, tagname_to_tagid):
    """
    function to return proper labels (for relevance column and for sectors column)
    """
    values_to_fill = row
    n_labels = len(tagname_to_tagid)
    row = [0] * n_labels
    for target_tmp in values_to_fill:
        row[tagname_to_tagid[target_tmp]] = 1
    return row


def custom_concat(row):
    sample = row[0]
    for array in row[1:]:
        sample = np.concatenate((sample, array), axis=0)
    return sample


def return_results_matrixes(VAL_PATH, INDEXES_PATH, PREDICTIONS_PATH):

    val_dataset = pd.read_csv(VAL_PATH)

    indexes = np.load(INDEXES_PATH)
    predictions_pillars = np.load(PREDICTIONS_PATH)

    val_dataset["target"] = val_dataset["target"].apply(lambda x: literal_eval(x))

    tagname_to_tagid = tagname_to_id(val_dataset["target"])

    val_dataset["target"] = val_dataset["target"].apply(lambda x: fill_column(x, tagname_to_tagid))

    preds = val_dataset[["entry_id"]]
    preds["predictions"] = preds["entry_id"].apply(lambda x: [0] * len(tagname_to_tagid))

    for i in range(len(indexes)):
        preds.loc[preds["entry_id"] == indexes[i], "predictions"] = preds.loc[
            preds["entry_id"] == indexes[i], "predictions"
        ].apply(lambda x: predictions_pillars[i, :])

    true_y = np.array([true_yss for true_yss in val_dataset["target"]])
    pred_y = np.array([true_yss for true_yss in preds["predictions"]])

    return true_y, pred_y, tagname_to_tagid
