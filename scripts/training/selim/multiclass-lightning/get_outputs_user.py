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
    columns = ['sectors', 
               'subpillars_2d', 
               'subpillars_1d', 
               'demographic_groups', 
               'affected_groups', 
               'specific_needs_groups']

    #cretae dict of ratio between probability of output and threshold
    ratio_proba_threshold = {}
    for column in columns:
        preds_column = test_probas[column]
        dict_keys = list(thresholds_dict[column].keys())
        nb_entries = len(dict_keys)

        returned_values_column = []
        for preds_sent in preds_column:
            dict_entry = {key:preds_sent[key]/thresholds_dict[column][key] for key in dict_keys }
            returned_values_column.append(dict_entry)
        ratio_proba_threshold[column] = returned_values_column

    predictions = {column:[] for column in columns}
    for i in range (nb_entries):

        # get the entries where the ratio is superior to 1 and put them in a dict {prediction:probability}
        for column in columns:
            preds_column = ratio_proba_threshold[column][i]
            preds_entry = {
                sub_tag:test_probas[column][i][sub_tag] for sub_tag in list(preds_column.keys()) if\
                        ratio_proba_threshold[column][i][sub_tag]>1
            }
            predictions[column].append(preds_entry)

        #postprocess 'subpillars_2d'
        if len(predictions['sectors'][i])>0 and len(predictions['subpillars_2d'][i])==0:
            predictions['subpillars_2d'][i] = {
                sub_tag:test_probas[column][i][sub_tag] for sub_tag in list(preds_column.keys()) if\
                        test_probas[column][i][sub_tag] == max(list(test_probas[column][i].values()))
            }

        #severity  predictions and output
        if 'Humanitarian Conditions' in predictions['subpillars_2d'][i]:
            pred_severity = {
                sub_tag:test_probas['severity'][i][sub_tag] for sub_tag in list(preds_column.keys()) if\
                        ratio_proba_threshold['severity'][i][sub_tag] == max(list(test_probas['severity'][i].values()))
            }

            predictions['severity'].append(pred_severity)
        else:
            predictions['severity'].append({})
            
    return predictions
