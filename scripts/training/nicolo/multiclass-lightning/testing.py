import numpy as np
import pandas as pd
from sklearn import metrics


def get_matrix(column_of_columns, tag_to_id, nb_subtags):
    matrix = [
        [1 if tag_to_id[i] in column else 0 for i in range(nb_subtags)]
        for column in column_of_columns
    ]
    return np.array(matrix)


def assess_performance(preds, groundtruth, subtags):
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
    tag_to_id = {i: subtags[i] for i in range(nb_subtags)}
    groundtruth_col = get_matrix(groundtruth, tag_to_id, nb_subtags)
    preds_col = get_matrix(preds, tag_to_id, nb_subtags)
    for j in range(groundtruth_col.shape[1]):
        preds_subtag = preds_col[:, j]
        groundtruth_subtag = groundtruth_col[:, j]
        results_subtag = {
            "macro_precision": np.round(
                metrics.precision_score(groundtruth_subtag, preds_subtag, average="macro"), 3
            ),
            "macro_recall": np.round(
                metrics.recall_score(groundtruth_subtag, preds_subtag, average="macro"), 3
            ),
            "macro_f1_score": np.round(
                metrics.f1_score(groundtruth_subtag, preds_subtag, average="macro"), 3
            ),
            "1_precision": np.round(
                metrics.precision_score(
                    groundtruth_subtag, preds_subtag, average="binary", pos_label=1
                ),
                3,
            ),
            "0_precision": np.round(
                metrics.precision_score(
                    groundtruth_subtag, preds_subtag, average="binary", pos_label=0
                ),
                3,
            ),
            "1_recall": np.round(
                metrics.recall_score(
                    groundtruth_subtag, preds_subtag, average="binary", pos_label=1
                ),
                3,
            ),
            "0_recall": np.round(
                metrics.recall_score(
                    groundtruth_subtag, preds_subtag, average="binary", pos_label=0
                ),
                3,
            ),
            "1_f1_score": np.round(
                metrics.f1_score(groundtruth_subtag, preds_subtag, average="binary", pos_label=1), 3
            ),
            "0_f1_score": np.round(
                metrics.f1_score(groundtruth_subtag, preds_subtag, average="binary", pos_label=0), 3
            ),
            "hamming_loss": np.round(metrics.hamming_loss(groundtruth_subtag, preds_subtag), 3),
            "zero_one_loss": np.round(metrics.zero_one_loss(groundtruth_subtag, preds_subtag), 3),
        }
        results_dict[subtags[j]] = results_subtag

    df_results = pd.DataFrame.from_dict(results_dict, orient="index")
    df_results.loc["mean"] = df_results.mean()

    return df_results
