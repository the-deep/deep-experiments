from ast import literal_eval
from sklearn.model_selection import train_test_split
import numpy as np

def clean_rows (row):
    """
    1) Apply litteral evaluation
    2) Drop values that are repeated multiple times in rows
    """
    return list(set(literal_eval(row)))

def get_subpillar_datasets (subpillar_name:str, dataset, n_synonym_augmenter=1, n_swap=1, perform_augmentation:bool=True):
    """
    1) keep rows where the sub-pillar name is contained in the column 'subpillars'
    2) keep only subpillar names in the column 'subpillar' (omit pillar name)
    """
    df = dataset[['entry_id', 'excerpt', 'pillars', 'subpillars', 'language']]
    df['subpillars'] = df.subpillars\
                        .apply(lambda x: list(filter(lambda y: subpillar_name in y, x)))\
                        .apply(lambda x: [y.split('->')[1] for y in (x)])

    df = df[df.subpillars.apply(lambda x: len(x)>0)]
    
    if perform_augmentation:
        train_data, test_data = train_test_split(df, test_size=0.3)
        return augment_data(train_data, n_synonym_augmenter, n_swap), test_data
    else: 
        return train_test_split(df, test_size=0.2)

def augment_data (df, n_synonym, n_swap):
    """
    1) Augment with synonym
    2) Apply swap on new (augmented with synonym) dataframe
    """
    if n_synonym:
        syn_aug_en = naw.SynonymAug(lang='eng', aug_min=3, aug_p=0.4)
        syn_aug_fr = naw.SynonymAug(lang='fra', aug_min=3, aug_p=0.4)
        syn_aug_es = naw.SynonymAug(lang='spa', aug_min=3, aug_p=0.4)

        en_syn = df[df.language=='en']
        fr_syn = df[df.language=='fr']
        es_syn = df[df.language=='es']

        en_syn.excerpt = en_syn.excerpt.apply(lambda x: syn_aug_en.augment(x, n=n_synonym))
        fr_syn.excerpt = fr_syn.excerpt.apply(lambda x: syn_aug_fr.augment(x, n=n_synonym))
        es_syn.excerpt = es_syn.excerpt.apply(lambda x: syn_aug_es.augment(x, n=n_synonym))

        whole_synoynm = pd.concat([en_syn, fr_syn, es_syn])
        
        for _, row in whole_synoynm.iterrows():
            excerpts = row.excerpt
            for i in range (0,n_synonym):
                row.excerpt = excerpts[i]
                df = df.append(row)

    if n_swap:
        swap = naw.RandomWordAug(action='swap', aug_min=3, aug_max=5)
        swap_df = df
        swap_df.excerpt = swap_df.excerpt.apply(lambda x: swap.augment(x, n=n_swap))

        for _, row in swap_df.iterrows():
            excerpts = row.excerpt
            for i in range (0,n_swap):
                row.excerpt = excerpts[i]
                df = df.append(row)


    return df

def tagname_to_id (target):
        tag_set = set()
        for tags_i in target:
            tag_set.update(tags_i)
        tagname_to_tagid = {tag:i for i, tag in enumerate(list(sorted(tag_set)))}
        return tagname_to_tagid


########################################### EVALUATION #########################################

def perfectEval(anonstring):
    try:
        ev = literal_eval(anonstring)
        return ev
    except ValueError:
        corrected = "\'" + anonstring + "\'"
        ev = literal_eval(corrected)
        return ev

def fill_column (row, n_labels, tagname_to_tagid):
    """
    function to return proper labels (for relevance column and for sectors column)
    """
    values_to_fill = row
    row = [0]*n_labels
    for target_tmp in values_to_fill:
        row[tagname_to_tagid[target_tmp]]=1
    return row

def custom_concat (row):
    sample = row[0]
    for array in row[1:]:
        sample = np.concatenate((sample, array), axis=0)
    return sample