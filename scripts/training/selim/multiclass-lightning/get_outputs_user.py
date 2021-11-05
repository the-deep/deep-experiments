multilabel_columns = [
    'sectors', 
    'subpillars_2d', 
    'subpillars_1d', 
    'age', 
    'gender', 
    'specific_needs_groups',
    ]

all_columns = multilabel_columns + ['severity']

def get_predictions(test_probas, thresholds_dict, nb_entries=100):  
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
    for column in all_columns + ['column_present']:
        preds_column = test_probas[column]
        dict_keys = list(thresholds_dict[column].keys())

        returned_values_column = []
        for preds_sent in preds_column:
            dict_entry = {key:preds_sent[key]/thresholds_dict[column][key] for key in dict_keys }
            returned_values_column.append(dict_entry)
        ratio_proba_threshold[column] = returned_values_column

    predictions = {column:[] for column in all_columns}
    for entry_nb in range (nb_entries):
        ratios_pos_neg_examples = ratio_proba_threshold['column_present'][entry_nb]
        preds_pos_neg_examples = [
            sub_tag for sub_tag in list(ratios_pos_neg_examples.keys()) if ratios_pos_neg_examples[sub_tag]>1
        ]
        # get the entries where the ratio is superior to 1 and put them in a dict {prediction:probability}
        for column in multilabel_columns:
            if column not in preds_pos_neg_examples:
                predictions[column].append([])
            else:
                preds_column = ratio_proba_threshold[column][entry_nb]
                preds_entry = [
                    sub_tag for sub_tag in list(preds_column.keys()) if preds_column[sub_tag]>1
                ]

                #postprocessing to keep only cross if more than one prediction
                """if column=='sectors' and len(preds_entry)>1:
                    preds_entry.append('Cross')"""

                if len(preds_entry)>1:
                    preds_entry = [
                        sub_tag for sub_tag in list(preds_column.keys())\
                            if preds_column[sub_tag]==max(list(preds_column.values()))
                    ]

                predictions[column].append(preds_entry)
                


        #postprocess 'subpillars_2d'
        """if len(predictions['sectors'][entry_nb])>0 and len(predictions['subpillars_2d'][entry_nb])==0:
            predictions['subpillars_2d'][entry_nb] = [
                sub_tag for sub_tag in list(preds_column.keys()) if\
                        test_probas[column][entry_nb][sub_tag] == max(list(test_probas[column][entry_nb].values()))
            ]

        if len(predictions['sectors'][entry_nb])==0 and len(predictions['subpillars_2d'][entry_nb])>0:
            predictions['subpillars_2d'][entry_nb] = []"""
            
        #severity  predictions and output
        if 'Humanitarian Conditions' in str(predictions['subpillars_2d'][entry_nb]):
            preds_column = ratio_proba_threshold['severity'][entry_nb]
            pred_severity = [
                sub_tag for sub_tag in list(preds_column.keys())\
                    if preds_column[sub_tag]==max(list(preds_column.values()))
            ]

            predictions['severity'].append(pred_severity)
        else:
            predictions['severity'].append([])
            
    return predictions