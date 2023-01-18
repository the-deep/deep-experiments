import torch
from typing import List, Dict, Tuple
import numpy as np
from copy import copy
import re
import random
import itertools
from collections import defaultdict
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# fix random state
random_state = 1234
random.seed(random_state)


def flatten(t: List[List]) -> List:
    """flatten list of lists"""
    return [item for sublist in t for item in sublist]


def keep_relevant_keys(input_dict: Dict, relevant_keys=List[str]):
    return {k: v for k, v in input_dict.items() if k in relevant_keys}


def fill_data_tensors(
    inputs_one_lead,
    attention_mask_one_lead,
    loss_mask_one_lead,
    pad_token_id: int,
    n_missing_elements: int,
    token_labels=None,
):
    """
    fill data tesors so they match the input length
    """
    added_input_ids_one_lead = torch.full((n_missing_elements,), pad_token_id)
    added_attention_mask_one_lead = torch.zeros(n_missing_elements)

    input_ids_one_lead = torch.cat([inputs_one_lead, added_input_ids_one_lead])
    attention_mask_one_lead = torch.cat(
        [attention_mask_one_lead, added_attention_mask_one_lead]
    )
    loss_mask_one_lead = torch.cat([loss_mask_one_lead, added_attention_mask_one_lead])

    if token_labels is not None:
        added_token_labels = torch.zeros((n_missing_elements, token_labels.shape[1]))
        token_labels_one_lead = torch.cat([token_labels, added_token_labels])
    else:
        token_labels_one_lead = None

    return (
        input_ids_one_lead,
        attention_mask_one_lead,
        loss_mask_one_lead,
        token_labels_one_lead,
    )


def beta_score(precision: float, recall: float, f_beta: float) -> float:
    """get beta score from precision and recall"""
    if precision * recall == 0:  # any of them is 0:
        return 0
    else:
        return (
            (1 + f_beta**2)
            * precision
            * recall
            / ((f_beta**2) * precision + recall)
        )


def _get_metrics(preds: List[int], groundtruth: List[int], f_beta=1):
    """
    metrics for one tag
    """
    precision, recall, f_score, _ = precision_recall_fscore_support(
        groundtruth, preds, average="binary", beta=f_beta
    )

    confusion_results = confusion_matrix(groundtruth, preds, labels=[0, 1])
    n_test_set_excerpts = sum(sum(confusion_results))
    accuracy = (confusion_results[0, 0] + confusion_results[1, 1]) / n_test_set_excerpts
    sensitivity = confusion_results[0, 0] / (
        confusion_results[0, 0] + confusion_results[0, 1]
    )
    specificity = confusion_results[1, 1] / (
        confusion_results[1, 0] + confusion_results[1, 1]
    )

    return {
        "precision": np.round(precision, 3),
        "recall": np.round(recall, 3),
        "f_score": np.round(f_score, 3),
        "accuracy": np.round(accuracy, 3),
        "sensitivity": np.round(sensitivity, 3),
        "specificity": np.round(specificity, 3),
    }


def _clean_str_for_logging(text: str):
    return re.sub("[^0-9a-zA-Z]+", "_", copy(text))


def _clean_results_for_logging(
    results: Dict[str, Dict[str, float]], prefix: str = ""
) -> Dict[str, float]:
    """clean names and prepare them for logging"""

    final_mlflow_outputs = {}
    for tagname, tagresults in results.items():
        for metric, score in tagresults.items():
            if type(score) is not str:
                mlflow_name = f"{prefix}_{tagname}_{metric}"
                mlflow_name = _clean_str_for_logging(mlflow_name)
                final_mlflow_outputs[mlflow_name] = score

    return final_mlflow_outputs


def get_full_outputs(
    gts_one_tag: List[int],
    best_predictions_cls: List[int],
    best_predictions_tokens: List[int],
    best_threshold_tokens: float,
    best_threshold_cls: float,
    fbeta: float,
):
    """
    each: List[Unique[0, 1]] of len(n_sentences)
    """
    full_outputs_one_tag = {}

    # hyperparameters
    full_outputs_one_tag[f"_optimal_threshold_tokens"] = best_threshold_tokens
    full_outputs_one_tag[f"_optimal_threshold_cls"] = best_threshold_cls

    # cls alone
    results_cls = _get_metrics(
        best_predictions_cls,
        gts_one_tag,
        fbeta,
    )["f_score"]

    # tokens alone
    results_tokens = _get_metrics(
        best_predictions_tokens,
        gts_one_tag,
        fbeta,
    )["f_score"]

    # intersection cls tokens
    intersection_tokens_cls = [
        1 if all([cls_pred, token_pred]) else 0
        for cls_pred, token_pred in zip(best_predictions_cls, best_predictions_tokens)
    ]
    results_intersection = _get_metrics(
        intersection_tokens_cls,
        gts_one_tag,
        fbeta,
    )["f_score"]

    # union cls tokens
    union_tokens_cls = [
        1 if any([cls_pred, token_pred]) else 0
        for cls_pred, token_pred in zip(best_predictions_cls, best_predictions_tokens)
    ]
    results_union = _get_metrics(
        union_tokens_cls,
        gts_one_tag,
        fbeta,
    )["f_score"]

    full_outputs_one_tag["f_score_results_cls_val"] = results_cls
    full_outputs_one_tag["f_score_results_tokens_val"] = results_tokens
    full_outputs_one_tag["f_score_results_intersection_val"] = results_intersection
    full_outputs_one_tag["f_score_results_union_val"] = results_union

    f_score_outputs = {
        "cls": results_cls,
        "tokens": results_tokens,
        "intersection": results_intersection,
        "union": results_union,
    }
    # optimal_setup = max(f_score_outputs, key=f_score_outputs.get)
    full_outputs_one_tag[
        "optimal_setup"
    ] = "union"  # force optimal setup to be the union of cls and tokens.

    return full_outputs_one_tag


def prepare_X_data(sentences: List[str], tokenizer):
    """
    loss_mask:
        0 for sep token id
        2 for cls token id
        1 for input ids
    """

    assert type(sentences) is list, "sentences inputs are not lists !"
    encoding = tokenizer(sentences, add_special_tokens=False)

    sentences_boundaries = []
    input_ids = []
    sentence_begin_offset = 0
    loss_mask = []

    for sentence_ids in encoding["input_ids"]:
        if len(sentence_ids) == 0:
            sentence_ids = [tokenizer.pad_token_id]

        sentence_end_offset = sentence_begin_offset

        # cls
        input_ids.append(tokenizer.cls_token_id)
        loss_mask.append(2)
        sentence_end_offset += 1

        # input ids
        input_ids.extend(sentence_ids)
        loss_mask.extend([1 for _ in range(len(sentence_ids))])
        sentence_end_offset += len(sentence_ids)

        sentences_boundaries.append(
            [sentence_begin_offset, sentence_end_offset]
        )  # because of the pythonic ways of selected ids in lists etc.

        # sep token id
        input_ids.append(tokenizer.sep_token_id)
        loss_mask.append(0)
        sentence_end_offset += 1

        # prepare for next one
        sentence_begin_offset = sentence_end_offset

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    loss_mask = torch.tensor(loss_mask, dtype=torch.long)

    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
    attention_mask[torch.where(input_ids != tokenizer.pad_token_id)] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sentences_boundaries": sentences_boundaries,
        "loss_mask": loss_mask,
    }


def custom_leads_stratified_splitting(
    project_ids: List[int],
    ratios: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1},
) -> Tuple[List[int], List[int], List[int]]:
    """
    custom function for stratified train val test splitting with respect to the project_id
    """

    train_ids = []
    val_ids = []
    test_ids = []

    initial_id = 0

    for proj_id, iter in itertools.groupby(
        sorted(project_ids)
    ):  # for project_id in unique_proj_ids:
        final_id = initial_id + len(list(iter))
        ids_lead = [i for i in range(initial_id, final_id)]
        initial_id = final_id

        train_val_ids_one_proj = random.sample(
            ids_lead, int(len(ids_lead) * (ratios["train"] + ratios["val"]))
        )
        test_ids_one_proj = list(set(ids_lead) - set(train_val_ids_one_proj))

        train_ids_one_proj = random.sample(
            train_val_ids_one_proj, int(len(train_val_ids_one_proj) * ratios["train"])
        )
        val_ids_one_proj = list(set(train_val_ids_one_proj) - set(train_ids_one_proj))

        train_ids.extend(train_ids_one_proj)
        val_ids.extend(val_ids_one_proj)
        test_ids.extend(test_ids_one_proj)

    return train_ids, val_ids, test_ids


def retrieve_sentences_probas_gt(
    all_leads_probas,
    all_leads_groundtruths,
    all_leads_sentences_offsets,
    leads_nb,
    omit_short_entries: bool = True,
):
    initial_sentence_ids = 0
    n_tags = all_leads_probas.shape[1]

    sentences_probas = [[] for _ in range(n_tags)]
    sentences_groundtruths = [[] for _ in range(n_tags)]

    for i in list(set(leads_nb)):

        one_lead_sentences_offsets = all_leads_sentences_offsets[i]

        for sentence_begin, sentence_end in one_lead_sentences_offsets:

            sent_len = sentence_end - sentence_begin
            final_sentences_ids = initial_sentence_ids + sent_len

            if (
                not omit_short_entries or sent_len > 3
            ):  # no highlightining sentences of 3 tokens or less
                probas_one_sent = all_leads_probas[
                    initial_sentence_ids:final_sentences_ids, :
                ]

                gt_one_sent = all_leads_groundtruths[
                    initial_sentence_ids, :
                ]  # take first id, made sure in sanity check that they re all the same

                for tag_idx in range(n_tags):
                    sentences_probas[tag_idx].append(probas_one_sent[:, tag_idx])
                    sentences_groundtruths[tag_idx].append(gt_one_sent[tag_idx].item())

            initial_sentence_ids = final_sentences_ids

    return sentences_probas, sentences_groundtruths


def _get_final_thresholds(
    all_leads_probas: torch.Tensor,
    all_leads_groundtruths: torch.Tensor,
    all_leads_sentences_offsets: torch.Tensor,
    tagname_to_id: Dict[str, int],
    leads_nb: List[int],
    fbeta: float,
):
    """
    ...
    """

    outputs = defaultdict()
    optimal_thresholds_cls = defaultdict()
    optimal_thresholds_tokens = defaultdict()

    # threshold lists for each tag
    mins = torch.min(all_leads_probas, dim=0).values.tolist()
    maxs = torch.max(all_leads_probas, dim=0).values.tolist()
    thresholds_possibilities = [
        np.round(np.linspace(min, max, 20), 3) for (min, max) in list(zip(mins, maxs))
    ]

    sentences_probas, sentences_groundtruths = retrieve_sentences_probas_gt(
        all_leads_probas,
        all_leads_groundtruths,
        all_leads_sentences_offsets,
        leads_nb,
        omit_short_entries=True,
    )

    for tag_name, tag_id in tagname_to_id.items():

        gts_one_tag = sentences_groundtruths[tag_id]
        probas_one_tag = sentences_probas[tag_id]
        thresholds_one_tag = thresholds_possibilities[tag_id][
            1:-1
        ]  # min and max proba predicted won't be the optimal thresholds

        best_f_score_cls = -1
        best_threshold_cls = -1
        best_predictions_cls = []

        best_f_score_tokens = -1
        best_threshold_tokens = -1
        best_predictions_tokens = []

        for one_threshold in thresholds_one_tag:

            # get ratios
            ratios_one_tag = [
                proba_per_tag_sentence / one_threshold
                for proba_per_tag_sentence in probas_one_tag
            ]

            # get cls predictions
            preds_per_sent_tag_cls = [
                1 if ratio_per_tag_sentence[0].item() >= 1 else 0
                for ratio_per_tag_sentence in ratios_one_tag
            ]

            # cls results
            results_per_threshold_tag_cls = _get_metrics(
                preds_per_sent_tag_cls,
                gts_one_tag,
                fbeta,
            )

            if results_per_threshold_tag_cls["f_score"] > best_f_score_cls:
                best_f_score_cls = results_per_threshold_tag_cls["f_score"]

                best_threshold_cls = one_threshold
                best_predictions_cls = preds_per_sent_tag_cls

            # tokens predictions, one threshold
            preds_per_sent_tag_tokens = [
                1 if ratio_per_tag_sentence[1:].mean().item() >= 1 else 0
                for ratio_per_tag_sentence in ratios_one_tag
            ]

            # tokens resuts, one threshold
            results_per_threshold_tokens = _get_metrics(
                preds_per_sent_tag_tokens,
                gts_one_tag,
                fbeta,
            )

            if results_per_threshold_tokens["f_score"] > best_f_score_tokens:
                best_f_score_tokens = results_per_threshold_tokens["f_score"]

                best_threshold_tokens = one_threshold
                best_predictions_tokens = preds_per_sent_tag_tokens

        # save best hyperparameters
        optimal_thresholds_cls[tag_name] = best_threshold_cls
        optimal_thresholds_tokens[tag_name] = best_threshold_tokens

        outputs[tag_name] = get_full_outputs(
            gts_one_tag,
            best_predictions_cls,
            best_predictions_tokens,
            best_threshold_tokens,
            best_threshold_cls,
            fbeta,
        )

    return outputs, optimal_thresholds_cls, optimal_thresholds_tokens


def hypertune_threshold(model, val_loader, fbeta):
    """
    main funtion for optimal threshold tuning
    """

    # len equals to the number of leads not to the number os rows
    all_leads_sentences_offsets = val_loader.dataset.data["sentences_boundaries"]

    # len equals to number of rows
    all_leads_probas = model._generate_probas(val_loader)

    all_leads_groundtruths = torch.cat(val_loader.dataset.data["token_labels"])
    all_leads_loss_masks = torch.cat(val_loader.dataset.data["loss_mask"])

    # keep only the backpropagated loss
    all_leads_groundtruths = all_leads_groundtruths[all_leads_loss_masks != 0]

    # from raw predictions to sentences

    (
        val_results,
        optimal_thresholds_cls,
        optimal_thresholds_tokens,
    ) = _get_final_thresholds(
        all_leads_probas,
        all_leads_groundtruths,
        all_leads_sentences_offsets,
        tagname_to_id=model.tagname_to_id,
        leads_nb=val_loader.dataset.data["leads_nb"],
        fbeta=fbeta,
    )

    return val_results, optimal_thresholds_cls, optimal_thresholds_tokens


def _create_y_data_matrix(
    excerpt_sentence_ids: List[Dict], n_sentences: int, tagname_to_tagid: Dict[str, int]
):
    """
    labels as matrix
    """
    labels = torch.zeros((n_sentences, len(tagname_to_tagid)))

    for one_sentence_ids_labels in excerpt_sentence_ids:
        sent_id = one_sentence_ids_labels["index"]
        sent_tags = one_sentence_ids_labels["tags"]

        for one_tag in sent_tags:
            labels[sent_id, tagname_to_tagid[one_tag]] = 1

    return labels


def _create_y_predictions_matrix(
    predictions: List[List[str]], n_sentences: int, tagname_to_tagid: Dict[str, int]
):
    predictions_matrix = torch.zeros((n_sentences, len(tagname_to_tagid)))
    for sent_id, predictions_one_sentence in enumerate(predictions):
        for one_predicted_tag in predictions_one_sentence:
            predictions_matrix[sent_id, tagname_to_tagid[one_predicted_tag]] = 1

    return predictions_matrix


def _generate_test_set_results(model, test_dataset, fbeta: float = 1):
    """
    generate predictions on test set  and get results.
    """

    # Generate predictions
    n_total_test_sentences = 0
    total_predictions, total_groudtruths = [], []

    for test_lead in test_dataset:
        # lead_id = test_lead["lead_id"]
        sentences = test_lead["sentences"]
        n_sentences_one_lead = len(sentences)
        n_total_test_sentences += n_sentences_one_lead

        excerpt_sentence_indices = test_lead["excerpt_sentence_indices"]
        groundtruth_matrix_one_lead = _create_y_data_matrix(
            excerpt_sentence_indices, n_sentences_one_lead, model.tagname_to_id
        )

        predictions_one_lead = model.get_highlights(sentences)
        assert (
            len(predictions_one_lead) == n_sentences_one_lead
        ), f"problem in test set sentences generation, len of 'predictions_one_lead': {len(predictions_one_lead['cls'])}, number of lead sentences: {n_sentences_one_lead}."

        predictions_matrix_one_lead = _create_y_predictions_matrix(
            predictions_one_lead, n_sentences_one_lead, model.tagname_to_id
        )

        total_predictions.append(predictions_matrix_one_lead)
        total_groudtruths.append(groundtruth_matrix_one_lead)

    total_predictions = torch.cat(total_predictions)
    total_groudtruths = torch.cat(total_groudtruths)

    ######### Generate results
    results_test_set = {}
    for tagname, tag_id in model.tagname_to_id.items():
        predictions_one_tag = total_predictions[:, tag_id].tolist()
        groundtruth_one_tag = total_groudtruths[:, tag_id].tolist()

        tag_metrics = _get_metrics(predictions_one_tag, groundtruth_one_tag, fbeta)
        results_test_set[tagname] = tag_metrics

    return results_test_set, n_total_test_sentences


def _get_results_df_from_dict(
    final_results: Dict[str, Dict[str, float]], proportions: Dict[str, Dict[str, float]]
):
    """
    input:
        final_results: Dict: {tagname: {metric: score}}
        proportions: {'cls': {tagname: proportion}, 'tokens': {tagname: proportion}}

    output: results as a dataframe and mean outputs of each tag
    """
    results_as_df = pd.DataFrame.from_dict(final_results, orient="index")
    metrics_list = list(results_as_df.columns)
    results_as_df["tag"] = results_as_df.index
    results_as_df.sort_values(by=["tag"], inplace=True, ascending=True)

    proportions_types = []
    for token_type, one_type_proportions in proportions.items():
        proportion_name = f"positive_examples_proportion_{token_type}"
        proportions_types.append(proportion_name)
        results_as_df[proportion_name] = [
            one_type_proportions[one_tag] for one_tag in results_as_df["tag"]
        ]

    ordered_columns = ["tag"] + metrics_list + proportions_types
    results_as_df = results_as_df[ordered_columns]

    return results_as_df
