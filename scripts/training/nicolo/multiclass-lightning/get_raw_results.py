def clean_col(preds):
    return [item for item in preds if item != "NOT_MAPPED"]


def compare_preds(groundtruth_col, preds_col):
    perfect_matches = 0
    one_missing = 0
    more_than_one_missing = 0
    one_false = 0
    more_than_one_false = 0
    len_col = len(groundtruth_col)
    for i in range(len_col):
        gt_sentence = set(clean_col(groundtruth_col[i]))
        pred_sentence = set(clean_col(preds_col[i]))
        union_tags = gt_sentence.union(pred_sentence)
        intersection_tags = gt_sentence.intersection(pred_sentence)
        if union_tags == intersection_tags:
            perfect_matches += 1
        else:
            union_minus_preds = union_tags - pred_sentence
            len_union_minus_preds = len(union_minus_preds)
            union_minus_gt = union_tags
            len_union_minus_gt = len(union_minus_gt)
            if len_union_minus_preds == 1:
                one_missing += 1
            else:
                more_than_one_missing += 1
            if len_union_minus_gt == 1:
                one_false += 1
            else:
                more_than_one_false += 1
    return {
        "proportion_perfect_matches": np.round(perfect_matches / len_col, 3),
        "proportion_one_missing": np.round(one_missing / len_col, 3),
        "proportion_more_than_one_missing": np.round(more_than_one_missing / len_col, 3),
        "proportion_one_false": np.round(one_false / len_col, 3),
        "proportion_more_than_one_false": np.round(more_than_one_false / len_col, 3),
    }
