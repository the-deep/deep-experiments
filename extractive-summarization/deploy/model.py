import pytorch_lighting as pl
from torch import nn
import torch
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder


class ExtractionModel(pl.LightingModule):
    def __init__(
        self,
        backbone,
        tokenizer,
        num_labels,
        token_loss_weight,
        loss_weights,
        slice_length,
        extra_context_length,
        n_separate_layers=None,
        separate_layer_groups=None,
    ):
        """
        Args:
            backbone: a string indicating the backbone model to use
            tokenizer: the used tokenizer
            num_labels: number of labels
            token_loss_weight: weight of the token-level loss e.g. 0.5 will
                result in even weighting of token-level and sentence-level loss
            loss_weights: contribution of the individual labels to the loss
            slice_length: length of the context that is fed into the model at
                once
            extra_context_length: length of prefix that will be fed to the
                model as additional context (without generating predictions)
            n_separate_layers: number of separate layers to use for different
                `separate_layer_groups`
            separate_layer_groups: list of lists of label indices indicating
                how to group labels into separate final layers
        """
        super().__init__()

        self.backbone = BertModel.from_pretrained(backbone)

        # split the backbone into separate layers
        if n_separate_layers is not None and n_separate_layers > 0:
            separate_layers_config = BertConfig.from_pretrained(backbone)
            separate_layers_config.num_hidden_layers = n_separate_layers

            if separate_layer_groups is None:
                separate_layer_groups = [[i] for i in range(num_labels)]

            self.separate_layers = nn.ModuleList(
                [BertEncoder(separate_layers_config) for _ in range(len(separate_layer_groups))]
            )
            self.separate_layer_groups = separate_layer_groups

            for lr in self.separate_layers:
                for i, layer in enumerate(lr.layer):
                    layer.load_state_dict(
                        self.backbone.encoder.layer[-n_separate_layers + i].state_dict()
                    )

            for _ in range(n_separate_layers):
                del self.backbone.encoder.layer[-1]
        else:
            self.separate_layers = None

        self.tokenizer = tokenizer
        self.out_projs = nn.ModuleList(
            [nn.Linear(self.backbone.config.hidden_size, 1) for _ in range(num_labels)]
        )
        self.num_labels = num_labels
        self.token_loss_weight = token_loss_weight
        self.slice_length = slice_length
        self.extra_context_length = extra_context_length
        self.loss_weights = (
            torch.tensor(loss_weights).unsqueeze(0) if loss_weights is not None else None
        )

    def training_step(self, batch, batch_idx):
        """
        batch: {
            "id": d["id"],
            "input_ids": d["input_ids"],
            "attention_mask": d["attention_mask"],
            "offset_mapping": d["offset_mapping"],
            "token_labels": d["token_labels"],
            "token_labels_mask": d["token_labels_mask"],
            "sentence_labels": d["sentence_labels"],
            "sentence_labels_mask": d["sentence_labels_mask"],
        }
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_labels_mask = batch["token_labels_mask"]
        token_labels = batch["token_labels"]
        length = input_ids.shape[0]

        n_steps = int(input_ids.shape[0] / self.slice_length)

        extra_context = torch.cat(
            [
                torch.full(
                    self.extra_context_length,
                    self.tokenizer.pad_token_id,
                    device=input_ids.device,
                ),
                input_ids[: length - self.extra_context_length],
            ],
            1,
        ).view(n_steps, self.slice_length)[:, : self.extra_context_length]

        input_ids = input_ids.view(n_steps, self.slice_length)
        attention_mask = attention_mask.view(n_steps, self.slice_length)

        # Adding extra context
        input_ids = torch.cat([extra_context, input_ids], 1)
        attention_mask = torch.cat([torch.ones_like(extra_context), attention_mask], 1)

        token_labels_mask = torch.cat(
            [
                torch.zeros_like(extra_context),
                token_labels_mask.view(n_steps, self.slice_length),
            ],
            1,
        )
        token_labels = torch.cat(
            [
                torch.zeros((*extra_context.shape, self.num_labels))
                .type_as(token_labels)
                .to(extra_context.device),
                token_labels.view((n_steps, self.slice_length, self.num_labels)),
            ],
            1,
        )

        # LOSS calculation
        logits = self(input_ids, attention_mask, token_labels)
        loss_weights = self.loss_weights.to(logits.device) if self.loss_weights is not None else 1.0

        loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        if token_labels_mask is not None:
            active_loss = token_labels_mask.view(-1) == 1
        else:
            active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        active_labels = token_labels.view(-1, self.num_labels)

        token_loss = (
            (loss_fct(active_logits, active_labels) * loss_weights).sum(-1)[active_loss].mean()
        )
        # TODO: Add sentence loss as well
        return token_loss

    def forward(
        self,
        input_ids,  # 2d array of shape(n_steps, slice_length)
        attention_mask,  # 2d array as above
        token_labels,  # 3d array
    ):
        logits = torch.zeros_like(token_labels)

        hidden_state = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state

        if self.separate_layers is None:
            for i in range(self.num_labels):
                logits[:, :, i] = self.out_projs[i](hidden_state)[..., 0]
        else:
            # the attention mask passed to `BertEncoder.forward` is an "extended" attention mask
            # (not the same as passed to `BertModel`) so we have to manually create it here
            # to call `BertEncoder.forward` directly.
            extended_attention_mask: torch.Tensor = self.backbone.get_extended_attention_mask(
                attention_mask, hidden_state.size()[:-1], self.backbone.device
            )

            for separate_layer_idx, label_group in enumerate(self.separate_layer_groups):
                h = self.separate_layers[separate_layer_idx](
                    hidden_state, attention_mask=extended_attention_mask
                ).last_hidden_state

                for i in label_group:
                    logits[:, :, i] = self.out_projs[i](h)[..., 0]
        return logits

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self(batch)
        return {"logits": output}
