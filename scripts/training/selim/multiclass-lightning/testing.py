import numpy as np
import pandas as pd
from sklearn import metrics

def get_matrix (column_of_columns, tag_to_id, nb_subtags):
    matrix = [[
        1 if tag_to_id[i] in column else 0 for i in range (nb_subtags)
    ] for column in column_of_columns]
    return np.array(matrix)

def assess_performance (preds, groundtruth, subtags):
    """
    INPUTS:
        preds: List[List[str]]: list containing list of predicted tags for each entry
        groundtruth: List[List[str]]: list containing list of true tags for each entry
        subtags: subtags list, sorted by alphabetical order 
    OUTPUTS:
        pd.DataFrame: rows: subtags, column: precision, recall, f1_score
    """
    results_dict = {}
    nb_subtags = len(subtags)
    tag_to_id = {i:subtags[i] for i in range (nb_subtags)}
    groundtruth_col = get_matrix( groundtruth, tag_to_id, nb_subtags)
    preds_col = get_matrix( preds, tag_to_id, nb_subtags)  
    for j in range(groundtruth_col.shape[1]):  
        preds_subtag = preds_col[:, j]
        groundtruth_subtag = groundtruth_col[:, j]
        results_subtag = {
            'precision': np.round(metrics.precision_score(groundtruth_subtag, preds_subtag, average='macro'), 3),
            'recall': np.round(metrics.recall_score(groundtruth_subtag, preds_subtag, average='macro'), 3),
            'f1_score': np.round(metrics.f1_score(groundtruth_subtag, preds_subtag, average='macro'), 3),
        }
        results_dict[subtags[j]] = results_subtag
        
    df_results = pd.DataFrame.from_dict(results_dict, orient='index')
    df_results.loc['mean'] = df_results.mean()
        
    return df_results