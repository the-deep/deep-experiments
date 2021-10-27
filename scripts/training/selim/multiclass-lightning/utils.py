from ast import literal_eval
import random
import numpy as np
import pandas as pd
import dill

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

def custom_stratified_train_test_split(df, positive_ids, ratios):
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
    positive_df = df[df.entry_id.isin(positive_ids)]
    
    unique_entries = list(np.unique(positive_df['target'].apply(str)))
    for entry in unique_entries:
        ids_entry = list(positive_df[positive_df.target.apply(str)==entry].entry_id.unique())

        train_ids_entry = random.sample(
            ids_entry, int(len(ids_entry) * ratios['train'])
            )
        val_ids_entry = list(
            set(ids_entry) - set(train_ids_entry)
        )

        train_ids.append(train_ids_entry)
        val_ids.append(val_ids_entry)
        
    return flatten(train_ids), flatten(val_ids)

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
    proportion_negative_examples_train_df: float = 0.1,
    proportion_negative_examples_val_df: float = 0.25,
    augment_numbers: bool = True):

    """
    main preprocessing function:
    1) get positive entries using the porportions of train test split
    2) add negative exapmles using the ratios defined in the training notebook

    NB: work with ids because the augmented sentences have the same entry_id as the original ones
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

    ratio_negative_positive_examples_val = get_ratio_neg_positive(
        proportion_negative_examples_val_df
    )

    #rename column to 'target' to be able to work on it generically
    dataset = df[
        ["entry_id", "excerpt", column_name, 'lead_id']
        ].rename(columns={column_name: "target"}).dropna().copy()
        
    dataset['target'] = dataset.target.apply(lambda x: clean_rows(x))

    if column_name=='sectors':
        dataset = dataset[dataset.target.apply(lambda x: len(x)<2)]

    entries_pos_dataset, all_negative_ids = get_negative_positive_examples (dataset)

    # If we want to deploy the model, we use all the positive data we have for training

    ratios = {
        'train':0.9,
        'val':0.1,
    }
    
        
    train_pos_entries, val_pos_entries =\
        custom_stratified_train_test_split(dataset, entries_pos_dataset, ratios)

    nb_train_pos_entries = len(train_pos_entries)
    nb_val_pos_entries = len(val_pos_entries)

    nb_val_neg_entries = int(ratio_negative_positive_examples_val * nb_val_pos_entries)
    nb_train_neg_entries = int(ratio_negative_positive_examples_train * nb_train_pos_entries)


    val_negative_ids = random.sample(all_negative_ids, nb_val_neg_entries)
    remaining_negative_ids = list(
        set(all_negative_ids) - set(val_negative_ids)
    )
    train_negative_ids = random.sample(remaining_negative_ids, nb_train_neg_entries)

    val_ids = list (set(val_negative_ids).union(set (val_pos_entries)))
    train_ids = list (set(train_negative_ids).union(set (train_pos_entries)))
    
    df_train = dataset[dataset.entry_id.isin(train_ids)].drop(columns=['lead_id'])
    df_val = dataset[dataset.entry_id.isin(val_ids)].drop(columns=['lead_id'])

    #perform numbers augmentation:
    #   - if we want to  
    #   - only for tags containing numbers (subpillars_1d and subpillars_2d)
    if 'subpillars_' in column_name and augment_numbers:
        df_train = augment_numbers_sentences(df_train)
        
    return df_train, df_val

def stats_train_test (
    df_train:pd.DataFrame,
    df_val: pd.DataFrame,
    column_name:str):
    """
    Sanity check of data (proportion negative examples)
    """
    def compute_ratio_negative_positive (df):
        nb_rows_negative = df[df.target.apply(lambda x: len(x)==0)].shape[0]
        if len(df)>0:
            return  np.round(nb_rows_negative / df.shape[0], 3)
        else:
            return 0 

    ratio_negative_positive = {
        f"ratio_negative_examples_train_{column_name}": compute_ratio_negative_positive (df_train),
        f"ratio_negative_examples_val_{column_name}": compute_ratio_negative_positive (df_val),
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
