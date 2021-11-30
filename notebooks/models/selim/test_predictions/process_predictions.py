from typing import List
from utils import merge_dicts, get_preds_entry, flatten

def get_predictions_prim_tags(
    ratio_proba_threshold, 
    thresholds_dict, 
    output_columns: List[str], 
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
            if 'sectors' not in preds_pos_neg_examples:
                predictions['sectors'].append([])
            else:
                preds_entry = get_preds_entry (
                    ratio_proba_threshold, 'sectors', entry_nb, return_at_least_one, ratio_nb)
                predictions['sectors'].append(preds_entry)
                
        if 'subpillars_2d' in output_columns:
            if 'subpillars_2d' not in preds_pos_neg_examples:
                predictions['subpillars_2d'].append([])
            else:
                preds_pillars_2d = get_preds_entry (
                    ratio_proba_threshold, 'pillars_2d', entry_nb, return_at_least_one, ratio_nb)
                returns_subpillars_2d = merge_dicts([
                    ratio_proba_threshold['subpillars_2d_part1'][entry_nb], 
                    ratio_proba_threshold['subpillars_2d_part2'][entry_nb]
                ])
                all_2d_preds = []
                for preds_pillars_tmp in preds_pillars_2d:
                    predssubpillars_tmp = {
                        key:value for key,value in returns_subpillars_2d.items() if key.split('->')[0] in preds_pillars_tmp
                    }
                    preds_entry = [
                        sub_tag for sub_tag in list(predssubpillars_tmp.keys()) if predssubpillars_tmp[sub_tag]>ratio_nb
                    ]
                    if len(preds_entry)==0:
                        preds_entry = [
                            sub_tag for sub_tag in list(predssubpillars_tmp.keys())\
                                if predssubpillars_tmp[sub_tag]==max(list(predssubpillars_tmp.values()))
                        ]
                    all_2d_preds.append(preds_entry)
                predictions['subpillars_2d'].append(flatten(all_2d_preds))

        if 'subpillars_1d' in output_columns:
            if 'subpillars_1d' not in preds_pos_neg_examples:
                predictions['subpillars_1d'].append([])
            else:
                preds_pillars_1d = get_preds_entry (
                    ratio_proba_threshold, 'pillars_1d', entry_nb, return_at_least_one, ratio_nb)
                returns_subpillars_1d = merge_dicts([
                    ratio_proba_threshold['subpillars_1d_part1'][entry_nb], 
                    ratio_proba_threshold['subpillars_1d_part2'][entry_nb],
                    ratio_proba_threshold['subpillars_1d_part3'][entry_nb]
                ])
                all_1d_preds = []
                for preds_pillars_tmp in preds_pillars_1d:
                    predssubpillars_tmp = {
                        key:value for key,value in returns_subpillars_1d.items() if key.split('->')[0] in preds_pillars_tmp
                    }
                    preds_entry = [
                        sub_tag for sub_tag in list(predssubpillars_tmp.keys()) if predssubpillars_tmp[sub_tag]>ratio_nb
                    ]
                    if len(preds_entry)==0:
                        preds_entry = [
                            sub_tag for sub_tag in list(predssubpillars_tmp.keys())\
                                if predssubpillars_tmp[sub_tag]==max(list(predssubpillars_tmp.values()))
                        ]
                    all_1d_preds.append(preds_entry)
                predictions['subpillars_1d'].append(flatten(all_1d_preds))

    return predictions