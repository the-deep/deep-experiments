from typing import List
from utils import merge_dicts, get_preds_entry, flatten

def get_predictions(
    ratio_proba_threshold, 
    thresholds_dict, 
    output_columns: List[str], 
    prim_tags: bool,
    subpillars_2d_cols: List[str] = None,
    subpillars_1d_cols: List[str] = None,
    nb_entries: int  = 100, 
    return_at_least_one: bool = False, 
    ratio_nb: int = 1):
    
    if prim_tags:
        return get_predictions_prim_tags(
                    ratio_proba_threshold, 
                    thresholds_dict, 
                    output_columns, 
                    subpillars_2d_cols=subpillars_2d_cols,
                    subpillars_1d_cols=subpillars_1d_cols,
                    nb_entries=nb_entries, 
                    return_at_least_one=return_at_least_one, 
                    ratio_nb=ratio_nb)
    else:
        return get_predictions_sec_tags(
                    ratio_proba_threshold, 
                    thresholds_dict, 
                    output_columns, 
                    nb_entries=nb_entries, 
                    return_at_least_one=return_at_least_one, 
                    ratio_nb=ratio_nb)
    

def get_predictions_sec_tags(
    ratio_proba_threshold, 
    thresholds_dict, 
    output_columns: List[str], 
    nb_entries: int  = 100, 
    return_at_least_one: bool = True, 
    ratio_nb: int = 1):
    
    training_columns = list(ratio_proba_threshold.keys())

    predictions = {column:[] for column in output_columns }
    for entry_nb in range (nb_entries):
        ratios_pos_neg_examples = ratio_proba_threshold['present_sec_tags'][entry_nb]
        preds_pos_neg_examples = [
            sub_tag for sub_tag in list(ratios_pos_neg_examples.keys()) if ratios_pos_neg_examples[sub_tag]>ratio_nb
        ]
        for column_name in training_columns:
            if column_name in output_columns:
                preds_entry = get_outputs_one_col (
                                ratio_proba_threshold,
                                column_name, 
                                preds_pos_neg_examples, 
                                entry_nb, 
                                return_at_least_one, 
                                ratio_nb)
                predictions[column_name].append(preds_entry)
            
    return predictions

    

def get_predictions_prim_tags(
    ratio_proba_threshold, 
    thresholds_dict, 
    output_columns: List[str], 
    subpillars_2d_cols: List[str] = None,
    subpillars_1d_cols: List[str] = None,
    nb_entries: int  = 100, 
    return_at_least_one: bool = True, 
    ratio_nb: int = 1):  
    """
    ratio_proba_threshold: ratio output between model probability output and threshold 
    thresholds_dict: dict with thresholds values
    output_columns: List primary tags of columns used for training
    nb_entries: nb entries in test set
    return_at_least_one: for pillars and subpillars: return at least one prediction or not
    ratio_nb: number above which we take the prediction
    """

    training_columns = list(ratio_proba_threshold.keys())

    predictions = {column:[] for column in output_columns }
    for entry_nb in range (nb_entries):
        ratios_pos_neg_examples = ratio_proba_threshold['present_prim_tags'][entry_nb]
        preds_pos_neg_examples = [
            sub_tag for sub_tag in list(ratios_pos_neg_examples.keys()) if ratios_pos_neg_examples[sub_tag]>ratio_nb
        ]
        
        if 'sectors' in output_columns:
            preds_entry = get_outputs_one_col (
                                ratio_proba_threshold,
                                'sectors', 
                                preds_pos_neg_examples, 
                                entry_nb, 
                                return_at_least_one, 
                                ratio_nb)
            predictions['sectors'].append(preds_entry)
                
        if 'subpillars_2d' in output_columns:
            if 'subpillars_2d' not in preds_pos_neg_examples:
                predictions['subpillars_2d'].append([])
            else:
                preds_pillars_2d = get_preds_entry (
                    ratio_proba_threshold['pillars_2d'][entry_nb], return_at_least_one, ratio_nb)
                returns_subpillars_2d = merge_dicts([
                    ratio_proba_threshold[term][entry_nb] for term in subpillars_2d_cols
                ])
                preds_entry = get_preds_entry (
                    returns_subpillars_2d, return_at_least_one, ratio_nb)
                
                preds_entry = [item for item in preds_entry if item.split('->')[0] in preds_pillars_2d]
                  
                predictions['subpillars_2d'].append(preds_entry)

        if 'subpillars_1d' in output_columns:
            if 'subpillars_1d' not in preds_pos_neg_examples:
                predictions['subpillars_1d'].append([])
            else:
                preds_pillars_1d = get_preds_entry (
                    ratio_proba_threshold['pillars_1d'][entry_nb], return_at_least_one, ratio_nb)
                returns_subpillars_1d = merge_dicts([
                    ratio_proba_threshold[term][entry_nb] for term in subpillars_1d_cols
                ])
                preds_entry = get_preds_entry (
                    returns_subpillars_1d, return_at_least_one, ratio_nb)
                
                preds_entry = [item for item in preds_entry if item.split('->')[0] in preds_pillars_1d]
                predictions['subpillars_1d'].append(preds_entry)

    return predictions

def get_outputs_one_col (
    ratio_proba_threshold,
    column_name, 
    preds_pos_neg_examples, 
    entry_nb, 
    return_at_least_one, 
    ratio_nb):
    
    if column_name not in preds_pos_neg_examples:
        return []
    else:
        preds_entry = get_preds_entry (
            ratio_proba_threshold[column_name][entry_nb], return_at_least_one, ratio_nb)
        return preds_entry