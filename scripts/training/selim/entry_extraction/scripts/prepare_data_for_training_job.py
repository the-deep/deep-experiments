from typing import Dict, List
from transformers import AutoTokenizer
from utils import custom_leads_stratified_splitting, prepare_X_data, keep_relevant_keys
import torch


class DataPreparation:
    def __init__(
        self,
        leads_dict: List[Dict],
        tagname_to_tagid: Dict[str, int],
        tokenizer_name_or_path: str,
    ):
        """
        need to ensure that the tokenizer used here is the same as the one used in the models.
        """
        self.original_data = leads_dict

        self.tagname_to_tagid = tagname_to_tagid

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self._encode_leads()

    def _get_label_vector(self, entry_tags: List[str]):
        target_ids = torch.zeros(len(self.tagname_to_tagid), dtype=torch.long)

        # every excerpt is relevant
        target_ids[self.tagname_to_tagid["is_relevant"]] = 1

        entry_tag_ids = [self.tagname_to_tagid[tag] for tag in entry_tags]
        target_ids[entry_tag_ids] = 1

        return target_ids

    def _create_y_data(self, excerpt_sentence_ids, input_ids, offset_mapping):
        # TODO: recheck all works

        # initiliaze token labels with emoty list, to be filled iteratively with lists of len n_labels
        n_tokens = input_ids.shape[0]
        token_labels = torch.zeros(
            (n_tokens, len(self.tagname_to_tagid)), dtype=torch.long
        )

        for (
            one_sent_tag
        ) in (
            excerpt_sentence_ids
        ):  # only the tagged sentences filled, everything else not needed
            entry_tags = one_sent_tag["tags"]
            sentence_begin, sentence_end = offset_mapping[one_sent_tag["index"]]

            token_labels[sentence_begin:sentence_end] = self._get_label_vector(
                entry_tags
            )

        return token_labels

    def _encode_one_lead(self, sample):

        """
        sample: one element from original text.
        """

        forward_data = prepare_X_data(sample["sentences"], self.tokenizer)

        input_ids = forward_data["input_ids"]
        attention_mask = forward_data["attention_mask"]
        sentences_boundaries = forward_data["sentences_boundaries"]
        loss_mask = forward_data["loss_mask"]

        token_labels = self._create_y_data(
            sample["excerpt_sentence_indices"], input_ids, sentences_boundaries
        )

        out = {
            "lead_id": sample["id"]["lead_id"],
            "project_id": sample["id"]["project_id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_labels": token_labels,
            "sentences_boundaries": sentences_boundaries,
            "loss_mask": loss_mask,
            "sentences": sample["sentences"],
        }

        return out

    def _encode_leads(self):

        """
        testing: bool for whether we are testing the ddf or if we want all the data
        save_split_dicts: bool: whether or not to save the data split to train val test

        - preprocessing function, to be run before training models
        - output is raw
            - no max length
            - still need to add context and to pad etc to lengths, which is done in the training,
                because different cdepending on the context we want to add
        """

        processed_data = [
            self._encode_one_lead(one_lead_data) for one_lead_data in self.original_data
        ]

        # stratified splitting: project-wise.
        project_ids = [lead["project_id"] for lead in processed_data]

        train_indices, val_indices, test_indices = custom_leads_stratified_splitting(
            project_ids
        )

        train = [
            keep_relevant_keys(
                processed_data[i],
                relevant_keys=[
                    "lead_id",
                    "input_ids",
                    "attention_mask",
                    "token_labels",
                    "sentences_boundaries",
                    "loss_mask",
                ],
            )
            for i in train_indices
        ]

        val = [
            keep_relevant_keys(
                processed_data[i],
                relevant_keys=[
                    "lead_id",
                    "input_ids",
                    "attention_mask",
                    "token_labels",
                    "sentences_boundaries",
                    "loss_mask",
                ],
            )
            for i in val_indices
        ]

        test = [
            keep_relevant_keys(
                processed_data[i],
                relevant_keys=[
                    "lead_id",
                    "input_ids",
                    "attention_mask",
                    "token_labels",
                    "sentences_boundaries",
                    "loss_mask",
                    "sentences",
                ],
            )
            for i in test_indices
        ]

        self.final_outputs = {
            "train": train,
            "val": val,
            "test": test
        }
