from utils import flatten
import numpy as np

multilabel_columns = [
    'sectors', 
    'pillars_2d',
    'pillars_1d',
    'subpillars_2d', 
    'subpillars_1d', 
    'demographic_groups', 
    'affected_groups', 
    'specific_needs_groups'
    ]

no_subpillar_columns = [
    'sectors',
    'demographic_groups', 
    'affected_groups', 
    'specific_needs_groups'
    ]

all_columns = [
    'sectors', 
    'subpillars_2d', 
    'subpillars_2d_postprocessed',
    'subpillars_1d', 
    'subpillars_1d_postprocessed',
    'demographic_groups', 
    'affected_groups', 
    'specific_needs_groups',
    'severity'
    ]

def postprocess_subpillars (ratios_pillars, ratios_subpillars, return_at_least_one=True):
    """
    postprocess subpillars using the pillars:
    1) keep only subpillars where the pillar was already predicted
    2)  if one pillar was predicted but no subpillar, we take the max ratio of the subbpillars
    for that pillar
    """
    
    results_subpillars = []

    ratios_subpillars_changed = {name: {} for name, _ in ratios_pillars.items()}
    for column_name, ratio in ratios_subpillars.items():
        split_column = column_name.split('->')
        ratios_subpillars_changed[split_column[0]].update({
            split_column[1]: ratio
        })

    positive_pillars = [
        column_name for column_name, ratio in ratios_pillars.items() if ratio >= 1
        ]
    if len (positive_pillars) == 0 and return_at_least_one:
        positive_pillars = [
        column_name for column_name, ratio in ratios_pillars.items() \
            if ratio == max(list(ratios_pillars.values()))
        ]

    if len (positive_pillars) == 0:
        return []
    
    for column_tmp in positive_pillars:
        dict_results_column = ratios_subpillars_changed[column_tmp]
        preds_column_tmp = [
            subtag for subtag, value in dict_results_column.items() if value >=1
        ]
        if len(preds_column_tmp)==0:
            preds_column_tmp = [
            subtag for subtag, value in dict_results_column.items() \
                if value == max(list(dict_results_column.values()))
        ]
        results_subpillars.append(preds_column_tmp)
        
    return flatten(results_subpillars)

def get_predictions(test_probas, thresholds_dict):  
    """
    test_probas structure example: {
        'sectors':[
            {'Nutrition': 0.032076582, 'Shelter': 0.06674846}, 
            {'Cross': 0.21885818,'Education': 0.07529669}
        ],
        'demographic_groups':[
            {'Children/Youth Female (5 to 17 years old)': 0.47860646, 'Children/Youth Male (5 to 17 years old)': 0.42560646},
            {'Children/Youth Male (5 to 17 years old)': 0.47860646, 'Infants/Toddlers (<5 years old)': 0.85}
        ],
        .
        .
        .
    }
    
    thresholds_dict structure example: {
        'sectors':{
            'Agriculture': 0.2,
            'Cross': 0.02,
            .
            .
        },
        'subpillars_2d':{
            'Humanitarian Conditions->Physical And Mental Well Being': 0.7,
            .
            .
        },
        .
        .     
    }
    
    First iteration:
    - create dict which has the same structure as 'test_probas': 
    - contains ratio probability of output divided by the threshold
    
    Second iteration:
    - keep ratios superior to 1 except:
        - for subpillars_2d: when no ratio is superior to 1 but there is at least one prediction for sectors
        - for severity (no threshold, just keep max if there is 'Humanitarian Conditions' in secondary tags outputs)
    """

    #create dict of ratio between probability of output and threshold
    ratio_proba_threshold = {}
    for column in multilabel_columns:
        preds_column = test_probas[column]
        dict_keys = list(thresholds_dict[column].keys())
        nb_entries = len([i for i in test_probas['sectors'] if i])

        returned_values_column = []
        for preds_sent in preds_column:
            dict_entry = {key:preds_sent[key]/thresholds_dict[column][key] for key in dict_keys }
            returned_values_column.append(dict_entry)
        ratio_proba_threshold[column] = returned_values_column

    predictions = {column:[] for column in all_columns}
    for i in range (nb_entries):

        # get the entries where the ratio is superior to 1 and put them in a dict {prediction:probability}
        for column in no_subpillar_columns:
            preds_column = ratio_proba_threshold[column][i]
            preds_entry = [
                sub_tag for sub_tag in list(preds_column.keys()) if ratio_proba_threshold[column][i][sub_tag]>1
            ]

            #postprocessing to keep only cross if more than one prediction
            if column=='sectors' and len(preds_entry)>1:
                preds_entry.append(['Cross'])

            predictions[column].append(list(np.unique(preds_entry)))

        preds_2d = postprocess_subpillars(
            ratio_proba_threshold['pillars_2d'][i],
            ratio_proba_threshold['subpillars_2d'][i],
            True
            )

        preds_1d = postprocess_subpillars(
            ratio_proba_threshold['pillars_1d'][i],
            ratio_proba_threshold['subpillars_1d'][i],
            False
            )

        predictions['subpillars_2d_postprocessed'].append(preds_2d)
        predictions['subpillars_1d_postprocessed'].append(preds_1d)

        #postprocess 'subpillars_2d'
        if len(predictions['sectors'][i])>0 and len(predictions['subpillars_2d'][i])==0:
            predictions['subpillars_2d'][i] = [
                sub_tag for sub_tag in list(preds_column.keys()) if\
                        test_probas[column][i][sub_tag] == max(list(test_probas[column][i].values()))
            ]

        if len(predictions['sectors'][i])==0 and len(predictions['subpillars_2d'][i])>0:
            predictions['subpillars_2d'][i] = []
            
        #severity  predictions and output
        if 'Humanitarian Conditions' in predictions['subpillars_2d'][i]:
            pred_severity = [
                sub_tag for sub_tag in list(preds_column.keys()) if\
                ratio_proba_threshold['severity'][i][sub_tag] == max(list(test_probas['severity'][i].values()))
            ]

            predictions['severity'].append(pred_severity)
        else:
            predictions['severity'].append({})
            
    return predictions
