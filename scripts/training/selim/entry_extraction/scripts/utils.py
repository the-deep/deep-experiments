import torch
from typing import List, Union, Dict


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


def prepare_X_data(sentences: List[str], tokenizer):

    encoding = tokenizer(
        sentences, add_special_tokens=False, return_offsets_mapping=True
    )

    input_ids = [tokenizer.cls_token_id]
    offset_mapping = [(0, 0)]

    prev_offset = 0

    for (sentence_ids, sentence_offsets) in zip(
        encoding["input_ids"], encoding["offset_mapping"]
    ):

        input_ids.extend(sentence_ids)
        input_ids.append(tokenizer.sep_token_id)

        offset_mapping.extend(
            [
                (prev_offset + start, prev_offset + end)
                for start, end in sentence_offsets
            ]
        )
        offset_mapping.append((0, 0))

        prev_offset += len(sentence_ids) + 1  # TODO: check this is right

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    offset_mapping = torch.tensor(offset_mapping, dtype=torch.long)

    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
    attention_mask[torch.where(input_ids != tokenizer.pad_token_id)] = 1

    return (input_ids, attention_mask, offset_mapping)
