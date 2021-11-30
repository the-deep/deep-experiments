from sklearn import metrics
import numpy as np
import pandas as pd

def get_matrix (column_of_columns, tag_to_id, nb_subtags):
    matrix = [[
        1 if tag_to_id[i] in column else 0 for i in range (nb_subtags)
    ] for column in column_of_columns]
    return np.array(matrix)

def assess_performance (preds, groundtruth, subtags, column_name):
    results_dict = {}
    nb_subtags = len(subtags)
    tag_to_id = {i:subtags[i] for i in range (nb_subtags)}
    groundtruth_col = get_matrix( groundtruth, tag_to_id, nb_subtags)
    preds_col = get_matrix( preds, tag_to_id, nb_subtags)  
    for j in range(groundtruth_col.shape[1]):  

        preds_subtag = preds_col[:, j]
        groundtruth_subtag = groundtruth_col[:, j]
        results_subtag = {
            'macro_precision': np.round(
                metrics.precision_score(groundtruth_subtag, preds_subtag, average='macro'), 3
            ),
            'macro_recall': np.round(
                metrics.recall_score(groundtruth_subtag, preds_subtag, average='macro'), 3
            ),
            'macro_f1_score': np.round(
                metrics.f1_score(groundtruth_subtag, preds_subtag, average='macro'), 3
            ),
            '1_precision': np.round(
                metrics.precision_score(groundtruth_subtag, preds_subtag, average='binary', pos_label=1), 3
            ),
            '0_precision': np.round(
                metrics.precision_score(groundtruth_subtag, preds_subtag, average='binary', pos_label=0), 3
            ),
            '1_recall': np.round(
                metrics.recall_score(groundtruth_subtag, preds_subtag, average='binary', pos_label=1), 3
            ),
            '0_recall': np.round(
                metrics.recall_score(groundtruth_subtag, preds_subtag, average='binary', pos_label=0), 3
            ),
            '1_f1_score': np.round(
                metrics.f1_score(groundtruth_subtag, preds_subtag, average='binary', pos_label=1), 3
            ),
            '0_f1_score': np.round(
                metrics.f1_score(groundtruth_subtag, preds_subtag, average='binary', pos_label=0), 3
            ),
            'hamming_loss': np.round(
                metrics.hamming_loss(groundtruth_subtag, preds_subtag), 3
            ),          
        }
        results_dict[subtags[j]] = results_subtag
        
    df_results = pd.DataFrame.from_dict(results_dict, orient='index')
    if 'NOT_MAPPED' in df_results.index:
        df_results = df_results.drop('NOT_MAPPED')
    df_results.loc[f'mean_{column_name}'] = df_results.mean()
        
    return df_results

def merge_dicts(x):
    z = x[0].copy()
    for i in range (1, len(x)):
        z.update(x[i])
        z.update(x[i])
    return z

def get_preds_entry (ratio_proba_threshold, column, entry_nb, return_at_least_one=True, ratio_nb=1):
    preds_column = ratio_proba_threshold[column][entry_nb]
    preds_entry = [
        sub_tag for sub_tag in list(preds_column.keys()) if preds_column[sub_tag]>1
    ]
    if return_at_least_one:
        if len(preds_entry)==0:
            preds_entry = [
                sub_tag for sub_tag in list(preds_column.keys())\
                    if preds_column[sub_tag]==max(list(preds_column.values()))
            ]
    return preds_entry

def flatten(t):
    return [item for sublist in t for item in sublist]

def clean_col(preds):
    return [item for item in preds if item!='NOT_MAPPED']

def compare_preds(groundtruth_col, preds_col):
    perfect_matches = 0
    at_leaset_one_missing = 0
    at_least_one_false = 0
    len_col = len(groundtruth_col)
    n_neg_examples = sum([1 if item==[] else 0 for item in groundtruth_col])
    n_neg_examples_truly_classified = 0
    for i in range (len_col):
        gt_sentence = set(clean_col(groundtruth_col[i]))
        pred_sentence = set(clean_col(preds_col[i]))
        union_tags = gt_sentence.union(pred_sentence)
        intersection_tags = gt_sentence.intersection(pred_sentence)
        if union_tags == intersection_tags:
            perfect_matches += 1
            if len(union_tags)==0:
                n_neg_examples_truly_classified += 1
        else:
            union_minus_preds = union_tags - pred_sentence
            len_union_minus_preds = len(union_minus_preds)
            union_minus_gt = union_tags
            len_union_minus_gt = len(union_minus_gt)
            if len_union_minus_preds >= 1:
                at_leaset_one_missing += 1
            if len_union_minus_gt >= 1:
                at_least_one_false += 1
    return {
        'proportion_perfect_matches': np.round(perfect_matches / len_col, 2),
        'proportion_negative_examples': np.round(n_neg_examples / len_col, 2),
        'proportion_neg_examples_truly_classified': np.round(n_neg_examples_truly_classified / n_neg_examples, 2) if n_neg_examples!=0 else 0,
        'proportion_at_least_one_false': np.round(at_least_one_false / len_col, 2),
        'proportion_at_leaset_one_missing': np.round(at_leaset_one_missing / len_col, 2)
    }