from torch import nn
import torch
from transformers import AutoModel
import math


class BasicModel(nn.Module):
    def __init__(
        self, backbone, tokenizer, num_labels, slice_length, extra_context_length
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone)
        self.tokenizer = tokenizer
        self.slice_length = slice_length
        self.extra_context_length = extra_context_length
        self.out_proj = nn.Linear(self.backbone.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None, labels_mask=None):
        out_dict = {}

        logits = []

        n_steps = int(input_ids.shape[1] / self.slice_length)
        batch_size, length = input_ids.shape

        extra_context = torch.cat(
            [
                torch.full(
                    (batch_size, self.extra_context_length),
                    self.tokenizer.pad_token_id,
                    device=input_ids.device,
                ),
                input_ids[:, : length - self.extra_context_length],
            ],
            1,
        ).view(batch_size * n_steps, self.slice_length)[:, : self.extra_context_length]

        input_ids = input_ids.view((batch_size * n_steps, self.slice_length))
        attention_mask = attention_mask.view((batch_size * n_steps, self.slice_length))

        # add extra context
        input_ids = torch.cat([extra_context, input_ids], 1)
        attention_mask = torch.cat([torch.ones_like(extra_context), attention_mask], 1)
        labels_mask = torch.cat(
            [
                torch.zeros_like(extra_context),
                labels_mask.view((batch_size * n_steps, self.slice_length)),
            ],
            1,
        )
        labels = torch.cat(
            [
                torch.zeros_like(extra_context),
                labels.view((batch_size * n_steps, self.slice_length)),
            ],
            1,
        )

        hidden_state = self.backbone(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state
        logits = self.out_proj(hidden_state)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if labels_mask is not None:
                active_loss = labels_mask.view(-1) == 1
            else:
                active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels),
            )

            out_dict["loss"] = loss_fct(active_logits, active_labels)

        out_dict["logits"] = logits[:, self.extra_context_length :].reshape(
            (batch_size, length, self.num_labels)
        )
        return out_dict


class RecurrentModel(nn.Module):
    def __init__(self, backbone, num_labels, slice_length):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone)
        self.slice_length = slice_length
        self.out_proj = nn.Linear(self.backbone.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        out_dict = {}

        logits = []

        cls_token = self.backbone.get_input_embeddings()(input_ids[:, 0])

        for i in range(math.ceil(input_ids.shape[1] / self.slice_length)):
            start, end = i * self.slice_length, (i + 1) * self.slice_length

            inputs_embeds = self.backbone.get_input_embeddings()(
                input_ids[:, start:end]
            )
            inputs_embeds[:, 0] = cls_token

            hidden_state = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask[:, start:end],
            ).last_hidden_state
            cls_token = hidden_state[:, 0]

            logits.append(self.out_proj(hidden_state))

        logits = torch.cat(logits, 1)
        out_dict["logits"] = logits

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels),
            )

            out_dict["loss"] = loss_fct(active_logits, active_labels)

        return out_dict


if __name__ == "__main__":
    input_ids = torch.zeros((1, 1024), dtype=torch.long)
    attention_mask = torch.ones((1, 1024), dtype=torch.long)
    labels = torch.ones((1, 1024), dtype=torch.long)

    model = RecurrentModel("microsoft/xtremedistil-l6-h256-uncased", 2)
    model(input_ids, attention_mask, labels)