from utils import prepare_X_data
from typing import Optional
import pytorch_lightning as pl
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, AutoModel
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from data import ExtractionDataset
from typing import List, Dict
from collections import defaultdict


class FocalLoss(nn.Module):
    # EPSILON is used to prevent infinity if some tag proportions are zero
    # valued. See in the constructor
    EPSILON = 1e-10

    def __init__(
        self,
        tag_token_proportions: Optional[torch.Tensor] = None,
        gamma: float = 2,
        proportions_pow: float = 1,
    ):
        """
        tag_token_proportions: Contains proportions of positive tokens for each tag.
            Its shape is 1 x num_tags

        returns non reduced focal loss.
        """

        weight = (
            torch.pow(
                1 / ((tag_token_proportions + FocalLoss.EPSILON) * 2), proportions_pow
            )
            if tag_token_proportions is not None
            else None
        )

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inp, target):
        ce_loss = F.binary_cross_entropy_with_logits(
            inp,
            # generally target is binary, so to be on the safe side convert to float64
            target.to(torch.float64),
            reduction="none",
        )
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma * ce_loss * self.weight).mean()
        return focal_loss


class EntryExtractor(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        slice_length: int,
        n_freezed_layers: int = 1,
    ):

        super().__init__()
        self.common_backbone = AutoModel.from_pretrained(model_name_or_path)
        self.common_backbone.encoder.layer = self.common_backbone.encoder.layer[:-1]
        self.slice_length = slice_length
        self.num_labels = num_labels

        # freeze embeddings
        for param in self.common_backbone.embeddings.parameters():
            param.requires_grad = False
        # freeze n_freezed_layers first layers
        if n_freezed_layers > 0:
            for layer in self.common_backbone.encoder.layer[:n_freezed_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.separate_layers = torch.nn.ModuleList(
            [
                AutoModel.from_pretrained(model_name_or_path).encoder.layer[-1]
                for _ in range(num_labels)
            ]
        )

        self.activation_function = nn.SELU()

    def forward(self, input_ids, attention_mask, loss_mask):
        logits = []

        hidden_state = self.common_backbone(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state

        for separate_layer_idx in range(self.num_labels):
            # not sure why we take the [..., 0] and not the mean, Benjamin does that and it works better.
            one_layer_output = self.activation_function(
                self.separate_layers[separate_layer_idx](hidden_state.clone())[0][
                    ..., 0
                ]
            )

            logits.append(one_layer_output)

        output = torch.stack(logits, dim=2)
        output = output[torch.where(loss_mask != 0)]

        return output


class TrainingExtractionModel(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str,
        tokenizer_name: str,
        tagname_to_tagid: Dict[str, int],
        tag_token_proportions: torch.Tensor,  # 1 x num_labels
        tag_cls_proportions: torch.Tensor,  # 1 x num_labels
        slice_length: int,
        extra_context_length: int,
        proportions_pow: float,
        tokens_focal_loss_gamma: float,
        cls_focal_loss_gamma: float,
        lr: float = 1e-4,
        adam_epsilon: float = 1e-7,
        weight_decay: float = 1e-2,
    ):
        """
        Args:
            backbone: a string indicating the backbone model to use
            tokenizer: the used tokenizer
            num_labels: number of labels
            token_loss_weight: weight of the token-level loss e.g. 0.5 will
                result in even weighting of token-level and sentence-level loss
            slice_length: length of the context that is fed into the model at
                once
            extra_context_length: length of prefix that will be fed to the
                model as additional context (without generating predictions)
        """
        super().__init__()

        self.tagname_to_id = tagname_to_tagid
        self.num_labels = len(tagname_to_tagid)
        self.entry_extraction_model = EntryExtractor(
            backbone_name, self.num_labels, slice_length
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.slice_length = slice_length
        self.extra_context_length = extra_context_length
        self.lr = lr
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay

        if torch.cuda.is_available():
            self.training_device = "cuda"
        else:
            self.training_device = "cpu"

        self.tag_token_proportions = tag_token_proportions.to(self.training_device)
        self.tag_cls_proportions = tag_cls_proportions.to(self.training_device)

        self.token_focal_loss = FocalLoss(
            self.tag_token_proportions, tokens_focal_loss_gamma, proportions_pow
        )
        self.cls_focal_loss = FocalLoss(
            self.tag_cls_proportions, cls_focal_loss_gamma, proportions_pow
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        loss_mask,
    ):
        output = self.entry_extraction_model(input_ids, attention_mask, loss_mask)
        return output

    def _operate_train_or_val_step(self, batch):
        """
        batch: {
            "input_ids": d["input_ids"],
            "attention_mask": d["attention_mask"],
            "token_labels": d["token_labels"],
            "loss_mask": d["loss_mask"]
        }
        """

        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["loss_mask"],
        )

        mask = batch["loss_mask"]
        backprop_mask = torch.where(mask != 0)

        labels = batch["token_labels"]
        backpropagated_labels = labels[backprop_mask]

        positive_loss_mask = mask[backprop_mask]
        cls_mask = torch.where(positive_loss_mask == 2)
        tokens_mask = torch.where(positive_loss_mask == 1)

        cls_logits = logits[cls_mask]
        cls_labels = backpropagated_labels[cls_mask]

        tokens_logits = logits[tokens_mask]
        tokens_labels = backpropagated_labels[tokens_mask]

        cls_loss = self.cls_focal_loss(cls_logits, cls_labels)
        tokens_loss = self.token_focal_loss(tokens_logits, tokens_labels)

        final_loss = cls_loss + tokens_loss

        """mask = batch["loss_mask"]
        important_labels = batch["token_labels"]
        important_labels = important_labels[torch.where(mask != 0)]

        final_loss = self._compute_loss(logits, important_labels)"""

        return final_loss

    def training_step(self, batch, batch_idx):

        train_loss = self._operate_train_or_val_step(batch)

        self.log(
            "train_loss",
            train_loss.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=False,
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self._operate_train_or_val_step(batch)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        return {"val_loss": val_loss}

    def configure_optimizers(self, *args, **kwargs):
        "Prepare optimizer and schedule (linear warmup and decay)"

        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon,
        )

        scheduler = StepLR(optimizer, gamma=0.4, step_size=1)

        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def _get_loaders(self, data, params, training_mode: bool):
        """
        get the dataloader from raw data
        """
        set = ExtractionDataset(
            dset=data,  # dict if training and raw text input text if test
            training_mode=training_mode,
            tokenizer=self.tokenizer,
            max_input_len=self.slice_length,
            extra_context_length=self.extra_context_length,
        )
        if training_mode:
            set.run_sanity_check()
        loader = DataLoader(set, **params, pin_memory=True)
        return loader


class LoggedExtractionModel(nn.Module):
    def __init__(self, trained_model) -> None:
        super().__init__()

        # get all values needed for inference into new class
        # new class used for logging
        self.trained_entry_extraction_model = trained_model.entry_extraction_model
        self.tokenizer = trained_model.tokenizer
        self.slice_length = trained_model.slice_length
        self.extra_context_length = trained_model.extra_context_length
        self.num_labels = trained_model.num_labels
        self.tagname_to_id = trained_model.tagname_to_id

        self.test_params = {"batch_size": 8, "shuffle": False, "num_workers": 0}

    def forward(
        self,
        input_ids,
        attention_mask,
        loss_mask,
    ):
        output = self.trained_entry_extraction_model(
            input_ids, attention_mask, loss_mask
        )
        return output

    def _get_loaders(self, data, params, training_mode: bool):
        """
        get the dataloader from raw data

        returned batch: {
            "input_ids": d["input_ids"],
            "attention_mask": d["attention_mask"],
            "loss_mask": d["loss_mask"]
        }
        """
        set = ExtractionDataset(
            dset=data,  # dict if training and raw text input text if test
            training_mode=training_mode,
            tokenizer=self.tokenizer,
            max_input_len=self.slice_length,
            extra_context_length=self.extra_context_length,
        )
        if training_mode:
            set.run_sanity_check()
        loader = DataLoader(set, **params, pin_memory=True)
        return loader

    def _generate_probas(self, data_loader):
        # can be on cpu
        if torch.cuda.is_available():
            testing_device = "cuda"
        else:
            testing_device = "cpu"

        self.to(testing_device)

        backbone_outputs = []

        with torch.no_grad():

            for batch in data_loader:

                logits = self(
                    batch["input_ids"].to(testing_device),
                    batch["attention_mask"].to(testing_device),
                    batch["loss_mask"].to(testing_device),
                ).cpu()

                # this is used if we want to recollate each lead together

                probabilities = torch.sigmoid(logits)

                backbone_outputs.append(probabilities)

        return torch.cat(backbone_outputs)

    def get_highlights(
        self, sentences: List[str], force_optimal_setup: str = None
    ) -> List[Dict[str, List[int]]]:

        """
        output: [{'cls': [predictions], 'tokens': [predictions]}, ...]

        one lead output only per call!
        """
        # optimal setups can be changed in case there is an issue
        if force_optimal_setup is not None:
            possible_setups = ["cls", "tokens", "union", "intersection"]
            assert (
                force_optimal_setup in possible_setups
            ), f"'force_optimal_setup' arg must be one of {possible_setups}, got {force_optimal_setup}"
            optimal_setup = {
                tagname: force_optimal_setup for tagname in self.tagname_to_id
            }
        else:
            optimal_setup = self.optimal_setups

        n_sentences = len(sentences)

        test_dset = prepare_X_data(sentences, self.tokenizer)

        test_loader = self._get_loaders(
            [test_dset], self.test_params, training_mode=False
        )  # only one lead

        probas = self._generate_probas(test_loader)

        # divide probas tensor(n_tokens, n_labels) by threshold tensor(n_labels)

        initial_sentence_ids = 0

        final_outputs = [[] for _ in range(n_sentences)]

        for sent_id, (sentence_begin, sentence_end) in enumerate(
            test_dset["sentences_boundaries"]
        ):

            sent_len = sentence_end - sentence_begin
            final_sentences_ids = initial_sentence_ids + sent_len

            if sent_len > 3:
                probas_one_sent = probas[initial_sentence_ids:final_sentences_ids, :]

                ratios_one_sent_cls = probas_one_sent / torch.tensor(
                    list(self.optimal_thresholds_cls.values())
                )
                ratios_one_sent_tokens = probas_one_sent / torch.tensor(
                    list(self.optimal_thresholds_tokens.values())
                )

                for tag_name, tag_id in self.tagname_to_id.items():
                    ratios_per_sent_tag_cls = ratios_one_sent_cls[0, tag_id].item()
                    ratios_per_sent_tag_tokens = (
                        ratios_one_sent_tokens[:, tag_id][1:].mean().item()
                    )

                    # cls vote
                    cls_vote = 1 if ratios_per_sent_tag_cls >= 1 else 0
                    tokens_vote = 1 if ratios_per_sent_tag_tokens >= 1 else 0

                    # return tagname if predicted value is 1.
                    if optimal_setup[tag_name] == "union" and any(
                        [cls_vote, tokens_vote]
                    ):
                        final_outputs[sent_id].append(tag_name)
                    elif optimal_setup[tag_name] == "intersection" and all(
                        [cls_vote, tokens_vote]
                    ):
                        final_outputs[sent_id].append(tag_name)
                    elif optimal_setup[tag_name] == "cls" and cls_vote:
                        final_outputs[sent_id].append(tag_name)
                    elif optimal_setup[tag_name] == "tokens" and tokens_vote:
                        final_outputs[sent_id].append(tag_name)

        return final_outputs
