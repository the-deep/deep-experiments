from collections import defaultdict
from typing import List, Dict, Union
from torch.utils.data import Dataset
from utils import fill_data_tensors
import torch


class ExtractionDataset(Dataset):
    def __init__(
        self,
        dset: Dict,
        training_mode: bool,
        tokenizer: str,
        max_input_len: int = 512,
        extra_context_length: int = 64,
    ):

        self.dset = dset
        self.training_mode = training_mode
        self.max_input_len = max_input_len
        self.extra_context_length = extra_context_length
        self.tokenizer = tokenizer

        self.data = self._prepare_data_for_forward_pass()

    def run_sanity_check(self):

        # len equals to the number of leads not to the number os rows
        all_leads_sentences_offsets = self.data["sentences_boundaries"]

        all_leads_groundtruths = torch.cat(self.data["token_labels"])
        all_leads_loss_masks = torch.cat(self.data["loss_mask"])

        # keep only the backpropagated loss
        all_leads_groundtruths = all_leads_groundtruths[all_leads_loss_masks == 1]

        # from raw predictions to sentences

        initial_sentence_ids = 0

        for i in list(set(self.data["leads_nb"])):
            original_lead_dset = self.dset[i]

            # input ids len checks with token losses etc.
            n_trainable_input_ids = len(
                [
                    token_id
                    for token_id in original_lead_dset["input_ids"]
                    if token_id
                    not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
                ]
            )

            one_lead_sentences_offsets = all_leads_sentences_offsets[i]
            sum_sentences_offsets = sum(
                [
                    final_offset - begin_offset
                    for begin_offset, final_offset in one_lead_sentences_offsets
                ]
            ).item()

            assert (
                n_trainable_input_ids == sum_sentences_offsets
            ), f"problem in input ids in lead_id={original_lead_dset['lead_id']}"

            for sentence_begin, sentence_end in one_lead_sentences_offsets:

                sent_len = sentence_end - sentence_begin
                final_sentences_ids = initial_sentence_ids + sent_len

                if sent_len > 2:  # no highlightining sentences of 2 tokens or less

                    one_sent_gt = all_leads_groundtruths[
                        initial_sentence_ids:final_sentences_ids
                    ]

                    # sanity check for groundtruths
                    assert (
                        torch.unique(one_sent_gt, dim=0).size(0) == 1
                    ), f"problem in groundtruths labels in lead_id={original_lead_dset['lead_id']}, nb={i}"

                initial_sentence_ids = final_sentences_ids

    def _prepare_data_for_forward_pass(self):
        """
        batch: same structure as in the '_operate_train_or_val_step' function.
        training: bool: whether we are training (the are present labels) or not (no loss computation needed)
        """

        final_outputs = defaultdict(list)

        n_leads = len(self.dset)

        for i in range(n_leads):

            ith_lead_data = self.dset[i]

            input_ids_one_lead = ith_lead_data["input_ids"].clone().detach().long()
            attention_mask_one_lead = (
                ith_lead_data["attention_mask"].clone().detach().long()
            )
            loss_mask_one_lead = ith_lead_data["loss_mask"].clone().detach().long()

            if self.training_mode:
                token_labels_one_lead = (
                    ith_lead_data["token_labels"].clone().detach().long()
                )

            if "sentences_boundaries" in ith_lead_data.keys():
                final_outputs["sentences_boundaries"].append(
                    ith_lead_data["sentences_boundaries"].clone().detach().long()
                )

            n_tokens_one_lead = len(input_ids_one_lead)

            if n_tokens_one_lead <= self.max_input_len:

                if n_tokens_one_lead < self.max_input_len:
                    (
                        input_ids_one_lead,
                        attention_mask_one_lead,
                        loss_mask_one_lead,
                        token_labels_one_lead,
                    ) = fill_data_tensors(
                        input_ids_one_lead,
                        attention_mask_one_lead,
                        loss_mask_one_lead,
                        self.tokenizer.pad_token_id,
                        self.max_input_len - n_tokens_one_lead,
                        token_labels_one_lead,
                    )

                final_outputs["input_ids"].append(input_ids_one_lead)
                final_outputs["attention_mask"].append(attention_mask_one_lead)
                final_outputs["leads_nb"].append(i)
                final_outputs["loss_mask"].append(loss_mask_one_lead)

                if self.training_mode:
                    final_outputs["token_labels"].append(token_labels_one_lead)

            else:
                initial_id = 0
                final_id = self.max_input_len

                step_size = self.max_input_len - 2 * self.extra_context_length

                while final_id < n_tokens_one_lead:

                    tmp_input_tensor = input_ids_one_lead[initial_id:final_id]
                    tmp_attention_mask_tensor = attention_mask_one_lead[
                        initial_id:final_id
                    ]

                    final_outputs["input_ids"].append(tmp_input_tensor)
                    final_outputs["attention_mask"].append(tmp_attention_mask_tensor)
                    final_outputs["leads_nb"].append(i)

                    tmp_loss_backprop_mask = loss_mask_one_lead[
                        initial_id:final_id
                    ].clone()

                    if initial_id == 0:  # initial tokens case
                        tmp_loss_backprop_mask[-self.extra_context_length :] = 0

                    else:  # middle of tokens case
                        tmp_loss_backprop_mask[-self.extra_context_length :] = 0
                        tmp_loss_backprop_mask[: self.extra_context_length] = 0

                    final_outputs["loss_mask"].append(tmp_loss_backprop_mask)

                    if self.training_mode:

                        final_outputs["token_labels"].append(
                            token_labels_one_lead[initial_id:final_id, :]
                        )

                    initial_id += step_size
                    final_id = initial_id + self.max_input_len

                # treating last slice
                tmp_input_tensor = input_ids_one_lead[-self.max_input_len :]
                tmp_attention_mask_tensor = attention_mask_one_lead[
                    -self.max_input_len :
                ]

                final_outputs["input_ids"].append(tmp_input_tensor)
                final_outputs["attention_mask"].append(tmp_attention_mask_tensor)
                final_outputs["leads_nb"].append(i)

                tmp_loss_backprop_mask = loss_mask_one_lead[
                    -self.max_input_len :
                ].clone()

                tmp_loss_backprop_mask[
                    : -(n_tokens_one_lead - initial_id - self.extra_context_length)
                ] = 0
                final_outputs["loss_mask"].append(tmp_loss_backprop_mask)

                if self.training_mode:
                    final_outputs["token_labels"].append(
                        token_labels_one_lead[-self.max_input_len :, :]
                    )

        return final_outputs

    def __getitem__(self, idx):
        out = {
            "input_ids": self.data["input_ids"][idx].clone().detach().long(),
            "attention_mask": self.data["attention_mask"][idx].clone().detach().long(),
            "loss_mask": self.data["loss_mask"][idx].clone().detach().long(),
        }

        if self.training_mode:
            out.update(
                {"token_labels": self.data["token_labels"][idx].clone().detach().long()}
            )

        return out

    def __len__(self):
        return len(self.data["leads_nb"])
