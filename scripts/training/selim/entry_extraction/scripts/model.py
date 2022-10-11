from utils import prepare_X_data, get_metric, get_label_vote_one_sentence, flatten
import pytorch_lightning as pl
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, AutoModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from data import ExtractionDataset
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict


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

    def forward(self, input_ids, attention_mask, loss_mask):
        logits = []

        hidden_state = self.common_backbone(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state

        for separate_layer_idx in range(self.num_labels):
            h = self.separate_layers[separate_layer_idx](hidden_state.clone())[0]

            cls_output = h[..., 0]
            mean_pooling = torch.mean(h, dim=-1)

            output = (cls_output + mean_pooling) / 2

            logits.append(output)

        output = torch.stack(logits, dim=2)
        output = output[torch.where(loss_mask == 1)]
        return output


class TrainingExtractionModel(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str,
        tokenizer_name: str,
        tagname_to_tagid: Dict[str, int],
        slice_length: int,
        extra_context_length: int,
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

    def _compute_loss(self, logits, groundtruth):

        token_loss = F.binary_cross_entropy_with_logits(
            logits, groundtruth.float(), reduction="mean"
        )

        return token_loss

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
        important_labels = batch["token_labels"]
        important_labels = important_labels[torch.where(mask == 1)]

        loss = self._compute_loss(logits, important_labels)

        return loss

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

        # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        scheduler = ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=1, threshold=1e-3
        )

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

        self.test_params = {"batch_size": 16, "shuffle": False, "num_workers": 0}

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
            for batch in tqdm(
                data_loader,
                total=len(data_loader.dataset) // data_loader.batch_size,
            ):

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
        """
        better to do sentences splitting on one side only, so no confusion.

        only function called in inference
        return predictions from raw input text

        1) split input text into sentences
        2) tokenizer and get off set for each one
        3) generate results for all tokens
        4) using offsets get relevant tokens for each sentence
        5) postprocessing to get results for each sentence.
            - postprocessing does not depend on the

        output: (n_words, n_labels), output[0, 1]: whether token 0 is relevant for label 1


        !!!!!! DESIGNED FOR ONE LEAD: SENTENCES: LIST OF ONE SPLIT LEAD.
        """

        final_outputs = []

        input_ids, attention_mask, offset_mapping = prepare_X_data(
            sentences, self.tokenizer
        )
        token_labels = None
        test_dset = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_labels": token_labels,
        }

        test_loader = self._get_loaders(
            test_dset,
            self.test_params,
        )

        probas = self._generate_probas(test_loader)

        ratios_probas_thresholds = probas / torch.tensor(
            list(self.optimal_thresholds.values())
        )  # divide probas tensor(n_tokens, n_labels) by threshold tensor(n_labels)

        n_sentences = len(sentences)

        for i in range(n_sentences):
            final_outputs_one_sentence = []
            # TODO: check the offset mapping
            token_begin_one_sent, token_end_one_sent = offset_mapping[i]

            ratios_one_sent = ratios_probas_thresholds[
                token_begin_one_sent:token_end_one_sent
            ]

            for tag_name, tag_id in self.tagname_to_id.items():
                label_votes_one_sent = get_label_vote_one_sentence(
                    ratios_one_sent, self.optimal_quantiles[tag_name]
                )

                final_outputs_one_sentence.append(label_votes_one_sent)

            final_outputs.append(final_outputs_one_sentence)

        return final_outputs

    def _get_final_thresholds(self, sentences_probas, sentences_groundtruths, fbeta):
        """
        ...
        """
        outputs = defaultdict(lambda: defaultdict())
        self.optimal_thresholds = defaultdict()
        self.optimal_quantiles = defaultdict()

        quantiles = np.linspace(0.1, 0.9, 9)  # 0.1 to 0.9 with step of 0.1

        # for probas_one_sentence, groundtruths_one_sentence in zip(pro)

        for tag_name, tag_id in self.tagname_to_id.items():

            probas_one_tag = [
                one_sentence_probas[:, tag_id]
                for one_sentence_probas in sentences_probas
            ]
            gts_one_tag = [
                one_sentence_groundtruths[tag_id]
                for one_sentence_groundtruths in sentences_groundtruths
            ]

            all_probas_one_tag = flatten(probas_one_tag)
            min_proba = np.round(min(all_probas_one_tag), 3)
            max_proba = np.round(max(all_probas_one_tag), 3)

            thresholds_one_tag = np.round(np.linspace(max_proba, min_proba, 21), 3)

            best_fbeta_score = 0
            best_recall = 0
            best_precision = 0
            best_threshold = 0
            best_quantile = 0

            for one_threshold in thresholds_one_tag:
                for one_quantile in quantiles:

                    preds_per_sent_tag = []

                    for preds_one_sent in probas_one_tag:
                        ratios_proba_threshold_one_sent = preds_one_sent / one_threshold
                        final_vote_one_sent = get_label_vote_one_sentence(
                            ratios_proba_threshold_one_sent, one_quantile
                        )
                        preds_per_sent_tag.append(final_vote_one_sent)

                    results_per_quantile_threshold_tag = get_metric(
                        preds_per_sent_tag,
                        gts_one_tag,
                        fbeta,
                    )

                    if (
                        results_per_quantile_threshold_tag["fbeta_score"]
                        > best_fbeta_score
                    ):
                        best_fbeta_score = results_per_quantile_threshold_tag[
                            "fbeta_score"
                        ]
                        best_recall = results_per_quantile_threshold_tag["recall"]
                        best_precision = results_per_quantile_threshold_tag["precision"]
                        best_quantile = one_quantile
                        best_threshold = one_threshold

            outputs[tag_name][f"{tag_name}_fbeta_score"] = np.round(best_fbeta_score, 2)
            outputs[tag_name][f"{tag_name}_precision"] = np.round(best_precision, 2)
            outputs[tag_name][f"{tag_name}_recall"] = np.round(best_recall, 2)
            outputs[tag_name][f"optimal_threshold_{tag_name}"] = best_threshold
            outputs[tag_name][f"optimal_quantile_{tag_name}"] = best_quantile

            self.optimal_thresholds[tag_name] = best_threshold
            self.optimal_quantiles[tag_name] = best_quantile

        return outputs

    def hypertune_threshold(self, val_loader, fbeta):
        lead_nbs = np.array(val_loader.dataset.data["leads_nb"])

        # len equals to the number of leads not to the number os rows
        all_leads_sentences_offsets = val_loader.dataset.data["sentences_boundaries"]

        # len equals to number of rows
        all_leads_probas = self._generate_probas(val_loader)
        all_leads_groundtruths = val_loader.dataset.data["token_labels"]

        # from raw predictions to sentences

        sentences_probas = []
        sentences_groundtruths = []

        for i in list(set(lead_nbs)):

            one_lead_ids = np.argwhere(lead_nbs == i).flatten()

            one_lead_probas = torch.cat(
                [
                    all_leads_probas[idx : idx + val_loader.batch_size]
                    for idx in one_lead_ids
                ]
            )
            one_lead_groundtruth = torch.cat(
                [all_leads_groundtruths[idx] for idx in one_lead_ids]
            )

            one_lead_sentences_offsets = all_leads_sentences_offsets[i]

            for sentence_begin, sentence_end in one_lead_sentences_offsets:
                if (
                    sentence_end > sentence_begin + 2
                ):  # no highlightining sentences of 2 tokens or less
                    sentences_probas.append(
                        one_lead_probas[sentence_begin:sentence_end]
                    )
                    sentences_groundtruths.append(one_lead_groundtruth[sentence_begin])

        final_results = self._get_final_thresholds(
            sentences_probas, sentences_groundtruths, fbeta
        )

        return final_results
