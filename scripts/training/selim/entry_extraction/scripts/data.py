from collections import defaultdict
from typing import List, Dict, Union
from torch.utils.data import Dataset
from utils import fill_data_tensors, create_loss_backprop_mask
import torch


class ExtractionDataset(Dataset):
    def __init__(
        self,
        dset: Union[Dict, str],  # dict if training and raw text input text if test
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

    def _prepare_data_for_forward_pass(self):

        """
        batch: same structure as in the '_operate_train_or_val_step' function.
        training: bool: whether we are training (the are present labels) or not (no loss computation needed)
        """

        if self.training_mode:
            input_ids = self.dset["input_ids"]
            attention_mask = self.dset["attention_mask"]
            token_labels = self.dset["token_labels"]
        else:
            input_ids, attention_mask = self.tokenizer(
                [self.dset], add_special_tokens=False, return_tensors="pt"
            )
            token_labels = None

        final_outputs = defaultdict(list)

        n_leads = len(input_ids)

        for i in range(n_leads):
            input_ids_one_lead = torch.tensor(input_ids[i], dtype=torch.long)
            attention_mask_one_lead = torch.tensor(attention_mask[i], dtype=torch.long)
            token_labels_one_lead = torch.tensor(token_labels[i], dtype=torch.long)

            n_tokens_one_lead = len(input_ids_one_lead)

            if n_tokens_one_lead <= self.max_input_len:

                if n_tokens_one_lead < self.max_input_len:
                    (
                        input_ids_one_lead,
                        attention_mask_one_lead,
                        token_labels_one_lead,
                    ) = fill_data_tensors(
                        input_ids_one_lead,
                        attention_mask_one_lead,
                        self.tokenizer.pad_token_id,
                        self.max_input_len - n_tokens_one_lead,
                        token_labels_one_lead,
                    )

                final_outputs["input_ids"].append(input_ids_one_lead)
                final_outputs["attention_mask"].append(attention_mask_one_lead)

                loss_backprop_mask = create_loss_backprop_mask(
                    attention_mask_one_lead,
                    input_ids_one_lead,
                    self.tokenizer.sep_token_id,
                    self.tokenizer.cls_token_id,
                )
                final_outputs["loss_mask"].append(loss_backprop_mask)

                if self.training_mode:
                    final_outputs["token_labels"].append(token_labels_one_lead)

            else:
                initial_id = 0
                final_id = self.max_input_len

                step_size = self.max_input_len - self.extra_context_length

                while final_id < n_tokens_one_lead:

                    tmp_input_tensor = input_ids_one_lead[initial_id:final_id]
                    tmp_attention_mask_tensor = attention_mask_one_lead[
                        initial_id:final_id
                    ]

                    final_outputs["input_ids"].append(tmp_input_tensor)
                    final_outputs["attention_mask"].append(tmp_attention_mask_tensor)

                    tmp_loss_backprop_mask = create_loss_backprop_mask(
                        tmp_attention_mask_tensor,
                        tmp_input_tensor,
                        self.tokenizer.sep_token_id,
                        self.tokenizer.cls_token_id,
                    )

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
                    final_id += step_size

                # treating last slice
                tmp_input_tensor = input_ids_one_lead[-self.max_input_len :]
                tmp_attention_mask_tensor = attention_mask_one_lead[
                    -self.max_input_len :
                ]

                final_outputs["input_ids"].append(tmp_input_tensor)
                final_outputs["attention_mask"].append(tmp_attention_mask_tensor)

                tmp_loss_backprop_mask = create_loss_backprop_mask(
                    tmp_attention_mask_tensor,
                    tmp_input_tensor,
                    self.tokenizer.sep_token_id,
                    self.tokenizer.cls_token_id,
                )
                tmp_loss_backprop_mask[: initial_id + self.extra_context_length] = 0
                final_outputs["loss_mask"].append(tmp_loss_backprop_mask)

                if self.training_mode:
                    final_outputs["token_labels"].append(
                        token_labels_one_lead[-self.max_input_len :, :]
                    )

        return final_outputs

    def __getitem__(self, idx):
        out = {
            "input_ids": torch.tensor(self.data["input_ids"][idx].clone().detach(), dtype=torch.long),
            "attention_mask": torch.tensor(
                self.data["attention_mask"][idx].clone().detach(), dtype=torch.long
            ),
            "loss_mask": torch.tensor(self.data["loss_mask"][idx].clone().detach(), dtype=torch.long),
        }

        if self.training_mode:
            out.update(
                {
                    "token_labels": torch.tensor(
                        self.data["token_labels"][idx].clone().detach(), dtype=torch.long
                    )
                }
            )

        return out

    def __len__(self):
        return len(self.dset)
