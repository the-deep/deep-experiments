from utils import (
    prepare_X_data,
    get_metric,
    retrieve_sentences_probas_gt,
    get_full_outputs,
)
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
import time


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

        scheduler = StepLR(optimizer, gamma=0.2, step_size=2)

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

    def get_highlights(self, sentences: List[str]):

        final_outputs = []

        test_dset = prepare_X_data(sentences, self.tokenizer)

        test_loader = self._get_loaders(
            [test_dset], self.test_params, training_mode=False
        )  # one lead

        probas = self._generate_probas(test_loader)

        # divide probas tensor(n_tokens, n_labels) by threshold tensor(n_labels)

        initial_sentence_ids = 0

        for sentence_begin, sentence_end in test_dset["sentences_boundaries"]:

            sent_len = sentence_end - sentence_begin
            final_sentences_ids = initial_sentence_ids + sent_len

            final_outputs_one_sentence = defaultdict(list)

            if sent_len > 3:
                probas_one_sent = probas[initial_sentence_ids:final_sentences_ids, :]

                ratios_one_sent_cls = probas_one_sent / torch.tensor(
                    list(self.optimal_thresholds_cls.values())
                )
                ratios_one_sent_tokens = probas_one_sent / torch.tensor(
                    list(self.optimal_thresholds_tokens.values())
                )

                for tag_name, tag_id in self.tagname_to_id.items():
                    ratios_per_sent_tag_cls = ratios_one_sent_cls[:, tag_id][0].item()
                    ratios_per_sent_tag_tokens = (
                        ratios_one_sent_tokens[:, tag_id][1:].mean().item()
                    )

                    # cls vote
                    cls_vote = 1 if ratios_per_sent_tag_cls >= 1 else 0
                    tokens_vote = 1 if ratios_per_sent_tag_tokens >= 1 else 0

                    # return tagname if prediited value is 1.
                    if cls_vote:
                        final_outputs_one_sentence["cls"].append(tag_name)
                    if tokens_vote:
                        final_outputs_one_sentence["tokens"].append(tag_name)

            final_outputs.append(final_outputs_one_sentence)

        return final_outputs

    def _get_final_thresholds(
        self,
        all_leads_probas: torch.Tensor,
        all_leads_groundtruths: torch.Tensor,
        all_leads_sentences_offsets: torch.Tensor,
        leads_nb: List[int],
        fbeta: float,
    ):
        """
        ...
        """

        outputs = defaultdict()
        self.optimal_thresholds_cls = defaultdict()
        self.optimal_thresholds_tokens = defaultdict()

        # threshold lists for each tag
        mins = torch.min(all_leads_probas, dim=0).values.tolist()
        maxs = torch.max(all_leads_probas, dim=0).values.tolist()
        thresholds_possibilities = [
            np.round(np.linspace(min, max, 10), 3)
            for (min, max) in list(zip(mins, maxs))
        ]

        sentences_probas, sentences_groundtruths = retrieve_sentences_probas_gt(
            all_leads_probas,
            all_leads_groundtruths,
            all_leads_sentences_offsets,
            leads_nb,
        )

        for tag_name, tag_id in self.tagname_to_id.items():

            gts_one_tag = sentences_groundtruths[tag_id]
            probas_one_tag = sentences_probas[tag_id]
            thresholds_one_tag = thresholds_possibilities[tag_id][
                1:-1
            ]  # min and max proba predicted won't be the optimal thresholds

            best_fbeta_score_cls = -1
            best_threshold_cls = -1
            best_predictions_cls = []

            best_fbeta_score_tokens = -1
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
                results_per_threshold_tag_cls = get_metric(
                    preds_per_sent_tag_cls,
                    gts_one_tag,
                    fbeta,
                )

                if results_per_threshold_tag_cls["fbeta_score"] > best_fbeta_score_cls:
                    best_fbeta_score_cls = results_per_threshold_tag_cls["fbeta_score"]

                    best_threshold_cls = one_threshold
                    best_predictions_cls = preds_per_sent_tag_cls

                # tokens predictions, one threshold
                preds_per_sent_tag_tokens = [
                    1 if ratio_per_tag_sentence[1:].mean().item() >= 1 else 0
                    for ratio_per_tag_sentence in ratios_one_tag
                ]

                # tokens resuts, one threshold
                results_per_threshold_tokens = get_metric(
                    preds_per_sent_tag_tokens,
                    gts_one_tag,
                    fbeta,
                )

                if (
                    results_per_threshold_tokens["fbeta_score"]
                    > best_fbeta_score_tokens
                ):
                    best_fbeta_score_tokens = results_per_threshold_tokens[
                        "fbeta_score"
                    ]

                    best_threshold_tokens = one_threshold
                    best_predictions_tokens = preds_per_sent_tag_tokens

            # save best hyperparameters
            self.optimal_thresholds_cls[tag_name] = best_threshold_cls
            self.optimal_thresholds_tokens[tag_name] = best_threshold_tokens

            outputs[tag_name] = get_full_outputs(
                tag_name,
                gts_one_tag,
                best_predictions_cls,
                best_predictions_tokens,
                best_threshold_tokens,
                best_threshold_cls,
                fbeta,
            )

        return outputs

    def hypertune_threshold(self, val_loader, fbeta):
        """
        ...
        """

        # len equals to the number of leads not to the number os rows
        all_leads_sentences_offsets = val_loader.dataset.data["sentences_boundaries"]
        n_sentences = sum([len(offsets) for offsets in all_leads_sentences_offsets])

        start_preds = time.process_time()
        # len equals to number of rows
        all_leads_probas = self._generate_probas(val_loader)
        end_preds = time.process_time()
        time_for_preds = end_preds - start_preds

        all_leads_groundtruths = torch.cat(val_loader.dataset.data["token_labels"])
        all_leads_loss_masks = torch.cat(val_loader.dataset.data["loss_mask"])

        # keep only the backpropagated loss
        all_leads_groundtruths = all_leads_groundtruths[all_leads_loss_masks != 0]

        start_thresholds = time.process_time()

        # from raw predictions to sentences

        final_results = self._get_final_thresholds(
            all_leads_probas,
            all_leads_groundtruths,
            all_leads_sentences_offsets,
            leads_nb=val_loader.dataset.data["leads_nb"],
            fbeta=fbeta,
        )

        end_thresholds = time.process_time()
        time_for_thresholds_tuning = end_thresholds - start_thresholds

        times = {
            "_time_batch_forward_per_sentence": round(time_for_preds / n_sentences, 5),
            "_time_threshold_calculation_per_sentence": round(
                time_for_thresholds_tuning / n_sentences, 2
            ),
        }

        return (final_results, times)
