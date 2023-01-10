import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from transformers import AdamW, AutoTokenizer, AutoModel
from data import ExcerptsDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np

from utils import (
    _flatten,
    get_tag_id_to_layer_id,
    get_first_level_ids,
    _get_parent_tags,
    _postprocess_predictions_one_excerpt,
)
from loss import FocalLoss
from pooling import Pooling


class TransformerArchitecture(torch.nn.Module):
    """
    base architecture, used for finetuning the transformer model.
    """

    def __init__(
        self,
        model_name_or_path,
        ids_each_level,
        dropout_rate: float,
        transformer_output_length: int,
        n_freezed_layers: int,
    ):
        super().__init__()
        self.ids_each_level = ids_each_level
        self.n_level0_ids = len(ids_each_level)
        self.n_heads = len(_flatten(self.ids_each_level))

        self.tag_id_to_layer_id = get_tag_id_to_layer_id(ids_each_level)

        self.common_backbone = AutoModel.from_pretrained(model_name_or_path)
        self.common_backbone.encoder.layer = self.common_backbone.encoder.layer[:-1]

        # freeze embeddings
        for param in self.common_backbone.embeddings.parameters():
            param.requires_grad = False
        # freeze n_freezed_layers first layers
        if n_freezed_layers > 0:
            for layer in self.common_backbone.encoder.layer[:n_freezed_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.pool = Pooling(
            word_embedding_dimension=transformer_output_length,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=True,
        )

        self.LayerNorm_specific_hidden = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(transformer_output_length * 2)
                for _ in range(self.n_level0_ids)
            ]
        )

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.specific_layer = torch.nn.ModuleList(
            [
                AutoModel.from_pretrained(model_name_or_path).encoder.layer[-1]
                for _ in range(self.n_level0_ids)
            ]
        )

        self.output_layer = torch.nn.ModuleList(
            [
                torch.nn.Linear(transformer_output_length * 2, len(id_one_level))
                for id_one_level in _flatten(ids_each_level)
            ]
        )

        self.activation_function = nn.SELU()

    def forward(self, inputs):

        explainability_bool = type(inputs) is tuple
        if explainability_bool:
            model_device = next(self.parameters()).device
            inputs = {
                "ids": inputs[0].to(model_device),
                "mask": inputs[1].to(model_device),
            }

        fith_layer_transformer_output = self.common_backbone(
            inputs["ids"],
            attention_mask=inputs["mask"],
        ).last_hidden_state

        encoder_outputs = [
            self.pool(
                {
                    "token_embeddings": self.specific_layer[i](
                        fith_layer_transformer_output.clone()
                    )[0],
                    "attention_mask": inputs["mask"],
                }
            )["sentence_embedding"]
            for i in range(self.n_level0_ids)
        ]

        if (
            "return_transformer_only" in inputs.keys()
            and inputs["return_transformer_only"]
        ):
            return torch.cat(encoder_outputs, dim=1)

        else:
            classification_heads = torch.cat(
                [
                    self.output_layer[tag_id](
                        self.LayerNorm_specific_hidden[self.tag_id_to_layer_id[tag_id]](
                            self.dropout(
                                self.activation_function(
                                    encoder_outputs[
                                        self.tag_id_to_layer_id[tag_id]
                                    ].clone()
                                )
                            )
                        )
                    )
                    for tag_id in range(self.n_heads)
                ],
                dim=1,
            )

            if not explainability_bool:
                return classification_heads
            else:
                return classification_heads.cpu()


class TrainingTransformer(pl.LightningModule):
    """
    pytorch lightning structure used for finetuning the trasformer.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        val_params: Dict[str, float],
        tagname_to_tagid,
        tags_proportions,
        loss_gamma,
        proportions_pow,
        n_freezed_layers: int,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-7,
        weight_decay: float = 0.1,
        dropout_rate: float = 0.3,
        max_len: int = 128,
        output_length: int = 384,
        training_device: str = "cuda",
        **kwargs,
    ):

        super().__init__()
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.transformer_output_length = output_length

        self.tagname_to_tagid = tagname_to_tagid
        self.ids_each_level = get_first_level_ids(self.tagname_to_tagid)
        self.max_len = max_len
        self.model = TransformerArchitecture(
            model_name_or_path,
            self.ids_each_level,
            dropout_rate,
            output_length,
            n_freezed_layers=n_freezed_layers,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.val_params = val_params

        self.training_device = training_device
        self.tags_proportions = tags_proportions

        self.loss = FocalLoss(
            tag_token_proportions=tags_proportions,
            gamma=loss_gamma,
            proportions_pow=proportions_pow,
            device=self.training_device,
        )

    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        train_loss = self.loss(outputs, batch["targets"])

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
        outputs = self(batch)
        val_loss = self.loss(outputs, batch["targets"])
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        return {"val_loss": val_loss}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon,
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def get_loaders(self, dataset, params):

        set = ExcerptsDataset(
            dataset, self.tagname_to_tagid, self.tokenizer, self.max_len
        )
        loader = DataLoader(set, **params, pin_memory=True)
        return loader


class LoggedTransformerModel(torch.nn.Module):
    """
    Logged transformers structure, done for space memory optimization
    Only contains needed varibles and functions for inference
    """

    def __init__(self, trained_model) -> None:
        super().__init__()
        self.trained_architecture = trained_model.model
        self.tokenizer = trained_model.tokenizer
        self.tagname_to_tagid = trained_model.tagname_to_tagid
        self.max_len = trained_model.max_len
        self.transformer_output_length = trained_model.transformer_output_length
        self.val_params = trained_model.val_params
        self.val_params["num_workers"] = 0
        self.parent_tags = _get_parent_tags(list(self.tagname_to_tagid.keys()))
        self.tags_proportions = trained_model.tags_proportions

        self.single_label_tags: List[List[str]] = [
            [
                "secondary_tags->severity->Critical issue",
                "secondary_tags->severity->Issue of concern",
                "secondary_tags->severity->Minor issue",
                "secondary_tags->severity->Severe issue",
            ],
            [
                "secondary_tags->Non displaced->Host",
                "secondary_tags->Non displaced->Non host",
            ],
            [
                "secondary_tags->Gender->All",
                "secondary_tags->Gender->Female",
                "secondary_tags->Gender->Male",
            ],
        ]

        self.all_postprocessed_labels = _flatten(self.single_label_tags)

    def forward(self, inputs):
        output = self.trained_architecture(inputs)
        return output

    def get_loaders(self, dataset, params):

        set = ExcerptsDataset(
            dataset, self.tagname_to_tagid, self.tokenizer, self.max_len
        )
        loader = DataLoader(set, **params, pin_memory=True)
        return loader

    def custom_predict(
        self,
        dataset,
        return_transformer_only: bool = True,
        hypertuning_threshold: bool = False,
    ):
        """
        1) get raw predictions
        2) postprocess them to output an output compatible with what we want in the inference
        # TODO: assertion in models to make sure args work together, clean everything
        """
        predictions_loader = self.get_loaders(
            dataset,
            self.val_params,
        )

        if torch.cuda.is_available():
            testing_device = "cuda"
        else:
            testing_device = "cpu"

        self.to(testing_device)

        logit_predictions, y_true = [], []

        with torch.no_grad():

            for batch in tqdm(
                predictions_loader,
                total=len(predictions_loader.dataset) // predictions_loader.batch_size,
            ):

                if hypertuning_threshold:
                    y_true.append(batch["targets"])

                logits = self(
                    {
                        "ids": batch["ids"].to(testing_device),
                        "mask": batch["mask"].to(testing_device),
                        "return_transformer_only": return_transformer_only,
                    }
                ).cpu()
                logit_predictions.append(logits)

        if len(logit_predictions) > 0:
            logit_predictions = torch.cat(logit_predictions, dim=0)
        else:
            logit_predictions = torch.tensor([])

        if return_transformer_only:
            return logit_predictions

        else:
            logit_predictions = torch.sigmoid(logit_predictions)

            if hypertuning_threshold:
                y_true = np.concatenate(y_true)
                return logit_predictions, y_true

            else:
                thresholds = np.array(list(self.optimal_thresholds.values()))
                final_predictions = logit_predictions.numpy() / thresholds

                outputs = [
                    {
                        tagname: final_predictions[i, tagid]
                        for tagname, tagid in self.tagname_to_tagid.items()
                    }
                    for i in range(logit_predictions.shape[0])
                ]
                # postprocess predictions

                return outputs

    def generate_test_predictions(
        self, entries: List[str], apply_postprocessing: bool = True
    ):

        model_outputs = self.custom_predict(
            entries,
            return_transformer_only=False,
            hypertuning_threshold=False,
        )

        if apply_postprocessing:
            postprocessed_outputs = [
                _postprocess_predictions_one_excerpt(
                    one_raw_pred,
                    self.all_postprocessed_labels,
                    self.single_label_tags,
                    self.parent_tags,
                )
                for one_raw_pred in model_outputs
            ]
        else:
            postprocessed_outputs = [
                [tagname for tagname, tagratio in one_raw_pred.items() if tagratio >= 1]
                for one_raw_pred in model_outputs
            ]
        return postprocessed_outputs
