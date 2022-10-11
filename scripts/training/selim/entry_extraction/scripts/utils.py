import torch
from typing import List, Union, Dict
import numpy as np
from sklearn import metrics


def flatten(t: List[List]) -> List:
    """flatten list of lists"""
    return [item for sublist in t for item in sublist]


def process_tag(tags: List[str], tag_section: str):
    if tag_section == "sectors":
        return [
            f"sectors->{one_tag}" for one_tag in tags if "NOT_MAPPED" not in one_tag
        ]
    else:  # subpillars
        return list(
            set(
                [
                    f"{tag_section.replace('sub', '')}->{one_tag.split('->')[0]}"
                    for one_tag in tags
                    if "NOT_MAPPED" not in one_tag
                ]
            )
        )


def keep_relevant_keys(input_dict: Dict, relevant_keys=List[str]):
    return {k: v for k, v in input_dict.items() if k in relevant_keys}


def create_loss_backprop_mask(attention_mask, input_ids, sep_token_id, cls_token_id):
    loss_backprop_mask = attention_mask.clone()
    loss_backprop_mask[torch.where(input_ids == sep_token_id)] = 0
    loss_backprop_mask[torch.where(input_ids == cls_token_id)] = 0
    return loss_backprop_mask


def fill_data_tensors(
    inputs_one_lead,
    attention_mask_one_lead,
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

    if token_labels is not None:
        added_token_labels = torch.zeros((n_missing_elements, token_labels.shape[1]))
        token_labels_one_lead = torch.cat([token_labels, added_token_labels])
    else:
        token_labels_one_lead = None

    return (input_ids_one_lead, attention_mask_one_lead, token_labels_one_lead)


def beta_score(precision: float, recall: float, f_beta: float) -> float:
    """get beta score from precision and recall"""
    return (1 + f_beta ** 2) * precision * recall / ((f_beta ** 2) * precision + recall)


def get_metric(preds, groundtruth, f_beta):

    precision = metrics.precision_score(
        groundtruth,
        preds,
        average="binary",
    )
    recall = metrics.recall_score(
        groundtruth,
        preds,
        average="binary",
    )
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

    preds_per_quantile_threshold_tag = (
        ratios_per_threshold_tag > quantile_per_threshold_tag
    )

    n_positive_votes = preds_per_quantile_threshold_tag.sum().item()
    n_negative_votes = len(ratios_per_threshold_tag) - n_positive_votes

    final_vote = 1 if n_positive_votes > n_negative_votes else 0

    return final_vote


def prepare_X_data(sentences: List[str], tokenizer):

    # TODO: check works

    sentences_boundaries = []

    encoding = tokenizer(sentences, add_special_tokens=False)

    input_ids = [tokenizer.cls_token_id]
    sentence_begin_offset = 1  # because the first input id is 'cls_token_id'

    for sentence_ids in encoding["input_ids"]:

        input_ids.extend(sentence_ids)

        sentence_end_offset = sentence_begin_offset + len(sentence_ids)
        sentences_boundaries.append(
            [sentence_begin_offset, sentence_end_offset]
        )  # because of the pythonic ways of seelcted ids in lists etc.

        sentence_begin_offset = sentence_end_offset

        input_ids.append(tokenizer.sep_token_id)
        sentence_begin_offset += 1  # because we add 'sep_token_id' between sentences

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    sentences_boundaries = torch.tensor(sentences_boundaries, dtype=torch.long)

    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
    attention_mask[torch.where(input_ids != tokenizer.pad_token_id)] = 1

    return (input_ids, attention_mask, sentences_boundaries)
