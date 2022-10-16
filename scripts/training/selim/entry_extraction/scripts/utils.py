from collections import defaultdict
import torch
from typing import List, Union, Dict, Tuple
import numpy as np
from sklearn import metrics
import re
from sklearn.model_selection import train_test_split


def flatten(t: List[List]) -> List:
    """flatten list of lists"""
    return [item for sublist in t for item in sublist]


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
            (1 + f_beta ** 2)
            * precision
            * recall
            / ((f_beta ** 2) * precision + recall)
        )


def get_metric(preds: List[int], groundtruth: List[int], f_beta: float):

    precision = metrics.precision_score(
        groundtruth, preds, average="binary", zero_division=0
    )
    recall = metrics.recall_score(groundtruth, preds, average="binary", zero_division=0)
    f_beta_score = beta_score(precision, recall, f_beta)
    return {
        "precision": np.round(precision, 3),
        "recall": np.round(recall, 3),
        "fbeta_score": np.round(f_beta_score, 3),
    }


def get_label_vote_one_sentence(
    ratios_per_threshold_tag: torch.Tensor, one_quantile: float
):
    quantile_per_threshold_tag = ratios_per_threshold_tag.quantile(one_quantile).item()

    final_vote = 1 if quantile_per_threshold_tag >= 1 else 0

    return final_vote


def clean_name_for_logging(dict_values: Dict[str, float]) -> Dict[str, float]:
    """clean names and prepare them for logging"""
    return {
        re.sub("[^0-9a-zA-Z]+", "_", name): value for name, value in dict_values.items()
    }


def prepare_X_data(sentences: List[str], tokenizer):

    sentences_boundaries = []

    assert type(sentences) is list, "sentences inputs are not lists !"
    encoding = tokenizer(sentences, add_special_tokens=False)

    input_ids = [tokenizer.cls_token_id]
    sentence_begin_offset = 1  # because the first input id is 'cls_token_id'
    loss_mask = [0]

    for sentence_ids in encoding["input_ids"]:
        if len(sentence_ids) == 0:
            sentence_ids = [tokenizer.pad_token_id]

        input_ids.extend(sentence_ids)

        sentence_end_offset = sentence_begin_offset + len(sentence_ids)
        sentences_boundaries.append(
            [sentence_begin_offset, sentence_end_offset]
        )  # because of the pythonic ways of seelcted ids in lists etc.
        loss_mask.extend([1 for _ in range(len(sentence_ids))])

        sentence_begin_offset = sentence_end_offset

        input_ids.append(tokenizer.sep_token_id)
        sentence_begin_offset += 1  # because we add 'sep_token_id' between sentences
        loss_mask.append(0)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    sentences_boundaries = torch.tensor(sentences_boundaries, dtype=torch.long)

    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
    attention_mask[torch.where(input_ids != tokenizer.pad_token_id)] = 1

    loss_mask = torch.tensor(loss_mask, dtype=torch.long)

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
    custom function for stratified train test splitting
    1) take unique sub-tags (example: ['Health'])
    2) For each unique subtag:
        i) take all indexes that have that specific subtag
        ii) split them randomly to train and test sets
    """
    # fix random state
    random_state = 1234

    train_ids = []
    val_ids = []
    test_ids = []

    unique_proj_ids = list(set(project_ids))

    for project_id in unique_proj_ids:

        ids_lead = [i for i, proj_id in enumerate(project_ids) if proj_id == project_id]
        n_leads_in_proj = len(ids_lead)

        train_val_indices, test_indices = train_test_split(
            ids_lead,
            train_size=int(n_leads_in_proj * (ratios["train"] + ratios["val"])),
            shuffle=True,
            random_state=random_state,
        )
        train_indices, val_indices = train_test_split(
            train_val_indices,
            train_size=int(n_leads_in_proj * ratios["train"]),
            random_state=random_state,
            shuffle=True,
        )

        train_ids.extend(train_indices)
        val_ids.extend(val_indices)
        test_ids.extend(test_indices)

    return train_ids, val_ids, test_ids


def retrieve_sentences_probas_gt(
    all_leads_probas, all_leads_groundtruths, all_leads_sentences_offsets, leads_nb
):  # TODO
    # get sentences

    initial_sentence_ids = 0
    n_tags = all_leads_probas.shape[1]

    sentences_probas = [[] for _ in range(n_tags)]
    sentences_groundtruths = [[] for _ in range(n_tags)]

    for i in list(set(leads_nb)):

        one_lead_sentences_offsets = all_leads_sentences_offsets[i]

        for sentence_begin, sentence_end in one_lead_sentences_offsets:

            sent_len = sentence_end - sentence_begin
            final_sentences_ids = initial_sentence_ids + sent_len

            if sent_len > 3:  # no highlightining sentences of 3 tokens or less
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
