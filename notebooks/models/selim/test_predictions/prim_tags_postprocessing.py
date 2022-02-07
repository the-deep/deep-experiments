pillars_1d_tags = ['Covid-19',
 'Casualties',
 'Context',
 'Displacement',
 'Humanitarian Access',
 'Shock/Event',
 'Information And Communication']

pillars_2d_tags = ['At Risk',
 'Priority Interventions',
 'Capacities & Response',
 'Humanitarian Conditions',
 'Impact',
 'Priority Needs']
 
def get_predictions_all(ratio_proba_threshold, 
    output_columns,
    pillars_2d,
    pillars_1d,
    nb_entries: int, 
    ratio_nb: int):
    
    predictions = {column:[] for column in output_columns }
    for entry_nb in range (nb_entries):
        returns_sectors = ratio_proba_threshold['sectors'][entry_nb] 
        preds_sectors = get_preds_entry (returns_sectors, False, ratio_nb)  
        predictions['sectors'].append(preds_sectors)
        
        returns_subpillars = ratio_proba_threshold['subpillars'][entry_nb] 
        
        subpillars_2d_tags = {
           key: value for key, value in returns_subpillars.items() if\
                key.split('->')[0] in pillars_2d
        }
        subpillars_1d_tags = {
           key: value for key, value in returns_subpillars.items() if\
                key.split('->')[0] in pillars_1d
        }
        if len(preds_sectors)==0:
            preds_2d = []
        else:
            preds_2d = get_preds_entry (subpillars_2d_tags, True, ratio_nb)
        
        predictions['subpillars_2d'].append(preds_2d)
        
        preds_1d = get_preds_entry (subpillars_1d_tags, False, ratio_nb)
        predictions['subpillars_1d'].append(preds_1d)

    return predictions

def get_preds_entry (preds_column, return_at_least_one=True, ratio_nb=1, return_only_one=False):
    preds_entry = [
        sub_tag for sub_tag in list(preds_column.keys()) if preds_column[sub_tag]>ratio_nb
    ]
    if return_only_one:
        preds_entry = [
            sub_tag for sub_tag in list(preds_column.keys())\
                if preds_column[sub_tag]==max(list(preds_column.values()))
        ]
    if return_at_least_one:
        if len(preds_entry)==0:
            preds_entry = [
                sub_tag for sub_tag in list(preds_column.keys())\
                    if preds_column[sub_tag]==max(list(preds_column.values()))
            ]
    return preds_entry

final_preds = get_predictions_all(
    doc['preds'], #raw predictions returned
    ['sectors', 'subpillars_2d', 'subpillars_1d'],
    pillars_2d=pillars_2d_tags,
    pillars_1d=pillars_1d_tags,
    nb_entries=n_preds, #total number of predictions to be postprocessed
    ratio_nb=1)

predictions_df = test_df[[
    'excerpt', 'entry_id'
]]
predictions_df['sectors'] = final_preds['sectors']
predictions_df['subpillars_2d'] = final_preds['subpillars_2d']
predictions_df['subpillars_1d'] = final_preds['subpillars_1d']

