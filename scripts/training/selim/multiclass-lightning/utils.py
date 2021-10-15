from ast import literal_eval
import random
from collections import Counter
import numpy as np
import pandas as pd

#from sklearn.model_selection import train_test_split
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

def custom_stratified_train_test_split(df, positive_ids, train_ratio:float = 0.8):
    """
    custom function for stratified train test splitting
    1) take unique sub-tags (example: ['Health'])
    2) For each unique subtag:
        i) take all indexes that have that specific subtag
        ii) split them ransomly to train and test sets

    NB: a bit time consuming ~ 950 entries/second in average
    """
    train_ids = []
    test_ids = []
    positive_df = df[df.entry_id.isin(positive_ids)]
    
    unique_entries = list(np.unique(positive_df['target'].apply(str)))
    for entry in unique_entries:
        ids_entry = list(positive_df[positive_df.target.apply(str)==entry].entry_id.unique())

        train_ids_entry = random.sample(
            ids_entry, int(len(ids_entry) * train_ratio)
            )
        test_ids_entry = list(
            set(ids_entry) - set(train_ids_entry)
        )

        train_ids.append(train_ids_entry)
        test_ids.append(test_ids_entry)
        
    return flatten(train_ids), flatten(test_ids)

def augment_numbers_sentences (df):
    """
    Generate new entries with changed numbers for tags containing numbers
    """
    def change_nb(word:str):
        try:
            word = int(word)
            changed_word = min(abs(random.randint(word-3, word+3)), 9)
            return changed_word
        except Exception:
            return word

    def augment_sent (txt:str):
        augmented_txt = ' '.join(
            ["".join([
                str(change_nb(c)) if c.isalnum() else c for c in word ]) for word in txt.split(' ')
                ]
        )
        return augmented_txt
        
    list_entries = list(np.unique(flatten(list(df.target))))
    entries_containing_numbers = [entry for entry in list_entries if 'Number' in entry]
    augmented_nb_df = df[df.target.apply(
            lambda x: np.any([tag in x for tag in entries_containing_numbers])
        )]
    augmented_nb_df['excerpt'] = augmented_nb_df.excerpt.apply(augment_sent)
    
    final_df = pd.concat([df, augmented_nb_df])
    
    return final_df

def get_negative_positive_examples (df: pd.DataFrame):

    """
    return the ids of entries containng negative samples
    1) filter leads that contain at least one postive example (at least one tagged entry)
    2) keep sentences with no tags
    """

    df_bis = df.copy()

    # Keep only unique values in pillars
    df_bis = df[['entry_id', 'target', 'lead_id']].copy()

    df_bis['count'] = df_bis.target.apply(lambda x: len(x))

    max_counts = df_bis[['lead_id', 'count']].groupby('lead_id', as_index=False).max()
    tagged_leads = max_counts[max_counts['count']>0].lead_id.tolist()

    all_negative_ids = df_bis[
        df_bis.lead_id.isin(tagged_leads) & df_bis.target.apply(lambda x: len(x)==0)
    ].entry_id.unique().tolist()
      
    ## POSITIVE ENTRIES:
    #list of positive entry_ids
    all_positive_ids = df_bis[df_bis.target.apply(lambda x: len(x) > 0)].entry_id.unique().tolist()

    return all_positive_ids, all_negative_ids



def preprocess_df(
    df: pd.DataFrame, 
    column_name: str, 
    deployment_mode: bool = False, 
    proportion_negative_examples_train_df: float = 0.1,
    proportion_negative_examples_test_df: float = 0.25,
    augment_numbers: bool = True,
    train_ratio: float = 0.8):

    """
    main preprocessing function:
    1) 
    """

    def get_ratio_neg_positive(ratio):
        """
        from total ratio of negative samples
        get ratio of negative /positive
        """
        return ratio / (1 - ratio)

    ratio_negative_positive_examples_train = get_ratio_neg_positive(
        proportion_negative_examples_train_df
    )

    ratio_negative_positive_examples_test = get_ratio_neg_positive(
        proportion_negative_examples_test_df
    )

    #rename column to 'target' to be able to work on it generically
    dataset = df[
        ["entry_id", "excerpt", column_name, 'lead_id']
        ].rename(columns={column_name: "target"}).dropna()
    dataset['target'] = dataset.target.apply(lambda x: clean_rows(x))

    if column_name=='sectors':
        dataset = dataset[dataset.target.apply(lambda x: len(x)<2)]

    entries_pos_dataset, all_negative_ids = get_negative_positive_examples (dataset)
    tot_len_pos_entries = len(entries_pos_dataset)

    # If we want to deploy the model, we use all the positive data we have for training
    if deployment_mode:
        train_pos_entries = entries_pos_dataset
        test_pos_entries = random.sample(
            entries_pos_dataset, int(tot_len_pos_entries * 0.05)
            )
        
    else:
        nb_train_pos_entries = int (train_ratio * tot_len_pos_entries)
        nb_test_pos_entries = tot_len_pos_entries - nb_train_pos_entries
        
        train_pos_entries, test_pos_entries = custom_stratified_train_test_split(dataset, entries_pos_dataset, train_ratio)

    nb_test_neg_entries = int(ratio_negative_positive_examples_test * nb_test_pos_entries)
    nb_train_neg_entries = int(ratio_negative_positive_examples_train * nb_train_pos_entries)

    test_negative_ids = random.sample(all_negative_ids, nb_test_neg_entries)
    remaining_negative_ids = list(
        set(all_negative_ids) - set(test_negative_ids)
    )
    train_negative_ids = random.sample(remaining_negative_ids, nb_train_neg_entries)

    """## NEGATIVE ENTRIES:
        nb_pos_entries_test_set = len(test_pos_entries)

        if len(all_negative_ids) > (nb_pos_entries_test_set * ratio_negative_positive_examples_test):
            test_negative_ids = random.sample(
                all_negative_ids, int(nb_pos_entries_test_set * ratio_negative_positive_examples_test)
                )
        else:
            test_negative_ids = all_negative_ids

        max_train_negative_ids = list(
            set(all_negative_ids) - set(test_negative_ids)
        )
    
    #If we want to include negative examples in the train set
    if bool(ratio_negative_positive_examples_train):

        number_of_negative_entries_added = int(
            len(train_pos_entries) * ratio_negative_positive_examples_train
            )

        if number_of_negative_entries_added >= len(max_train_negative_ids):
            train_negative_ids = max_train_negative_ids
        else:
            train_negative_ids = random.sample(max_train_negative_ids, number_of_negative_entries_added)

    else:
        train_negative_ids = []"""

    test_ids = list (set(test_negative_ids).union(set (test_pos_entries)))
    train_ids = list (set(train_negative_ids).union(set (train_pos_entries)))
    
    df_train = dataset[dataset.entry_id.isin(train_ids)].drop(columns=['lead_id'])
    df_test = dataset[dataset.entry_id.isin(test_ids)].drop(columns=['lead_id'])

    #perform numbers augmentation:
    #   - if we want to  
    #   - only for tags containing numbers (subpillars_1d and subpillars_2d)
    if 'subpillars_' in column_name and augment_numbers:
        df_train = augment_numbers_sentences(df_train)
        
    return df_train, df_test

def stats_train_test (df_train:pd.DataFrame, df_test:pd.DataFrame, column_name:str):
    """
    Sanity check of data (proportion negative examples)
    """
    def compute_ratio_negative_positive (df):
        nb_rows_negative = df[df.target.apply(lambda x: len(x)==0)].shape[0]
        return  np.round(nb_rows_negative / df.shape[0], 3)

    ratio_negative_positive = {
        f"ratio_negative_examples_train_{column_name}": compute_ratio_negative_positive (df_train),
        f"ratio_negative_examples_test_{column_name}": compute_ratio_negative_positive (df_test),
    }

    return ratio_negative_positive

def get_full_list_entries (df, tag):
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

