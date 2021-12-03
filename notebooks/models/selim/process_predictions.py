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
                    ratio_proba_threshold['sectors'][entry_nb], return_at_least_one, ratio_nb)
                predictions['sectors'].append(preds_entry)
                
        if 'subpillars_2d' in output_columns:
            if 'subpillars_2d' not in preds_pos_neg_examples:
                predictions['subpillars_2d'].append([])
            else:
                preds_pillars_2d = get_preds_entry (
                    ratio_proba_threshold['pillars_2d'][entry_nb], return_at_least_one, ratio_nb)
                returns_subpillars_2d = merge_dicts([
                    ratio_proba_threshold['impact_capresp_humcond'][entry_nb], 
                    ratio_proba_threshold['need_intervention_risk'][entry_nb]
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
                    ratio_proba_threshold['context_covid'][entry_nb], 
                    ratio_proba_threshold['displacement_shockevent'][entry_nb],
                    ratio_proba_threshold['access_infcom_casualities'][entry_nb]
                ])
                preds_entry = get_preds_entry (
                    returns_subpillars_1d, return_at_least_one, ratio_nb)
                
                preds_entry = [item for item in preds_entry if item.split('->')[0] in preds_pillars_1d]
                predictions['subpillars_1d'].append(preds_entry)

    return predictions

