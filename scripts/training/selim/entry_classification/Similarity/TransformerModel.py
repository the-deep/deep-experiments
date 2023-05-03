import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from transformers import AdamW, AutoTokenizer, AutoModel
from data import ExcerptsDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
import random

# from sklearn.metrics.pairwise import euclidean_distances

from utils import (
    _flatten,
    get_tag_id_to_layer_id,
    get_first_level_ids,
    _get_parent_tags,
    _postprocess_predictions_one_excerpt,
)
from loss import FocalLoss
from pooling import Pooling


### ARCHITECTURES ###


class TransformerArchitecture(torch.nn.Module):
    """
    base architecture, used for finetuning the transformer model.
    """

    def __init__(
        self,
        model_name_or_path,
        ids_each_level,
        tag_names: List[str],
        tokenizer,
        device,
        dropout_rate: float,
        transformer_output_length: int,
        n_freezed_layers: int,
    ):
        super().__init__()
        self.ids_each_level = ids_each_level
        self.n_level0_ids = len(ids_each_level)
        self.n_heads = len(_flatten(self.ids_each_level))
        self.tag_id_to_layer_id = get_tag_id_to_layer_id(ids_each_level)
        self.model_device = device
        # print(self.tag_id_to_layer_id)
        self.tag_names = [
            " ".join(name.split("->")[1:]).replace("_", " ") for name in tag_names
        ]
        self.tokenizer = tokenizer
        self.common_backbone = AutoModel.from_pretrained(model_name_or_path).to(
            self.model_device
        )
        self.pool = Pooling(
            word_embedding_dimension=transformer_output_length,
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=True,
        )

        # freeze embeddings
        for param in self.common_backbone.embeddings.parameters():
            param.requires_grad = False
        # freeze n_freezed_layers first layers
        if n_freezed_layers > 0:
            for layer in self.common_backbone.encoder.layer[:n_freezed_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.transformer_output_length = transformer_output_length
        self.final_output_length = transformer_output_length

        self.dropout = torch.nn.Dropout(dropout_rate)

        # self.specific_layer = torch.nn.ModuleList(
        #     [
        #         AutoModel.from_pretrained(model_name_or_path).encoder.layer[-1]
        #         for _ in range(self.n_level0_ids)
        #     ]
        # )
        self.activation_function = nn.SELU()

        self.n_tags = len(self.tag_names)
        midlayer_len = self.final_output_length // 2

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.final_output_length, midlayer_len),
            torch.nn.Dropout(dropout_rate),
            self.activation_function,
            torch.nn.LayerNorm(midlayer_len),
            torch.nn.Linear(midlayer_len, self.n_tags),
        )

    def forward(self, inputs, return_embeddings: bool):
        explainability_bool = type(inputs) is tuple
        if explainability_bool:
            model_device = next(self.parameters()).device
            inputs = {
                "ids": inputs[0].to(model_device),
                "mask": inputs[1].to(model_device),
            }

        encoder_outputs = self.common_backbone(
            inputs["ids"],
            attention_mask=inputs["mask"],
        ).last_hidden_state
        encoder_outputs = self.pool(
            {
                "token_embeddings": encoder_outputs,
                "attention_mask": inputs["mask"],
            }
        )["sentence_embedding"]

        if return_embeddings:
            return encoder_outputs  # torch.cat(encoder_outputs, dim=1)

        else:
            classification_heads = self.output_layer(encoder_outputs)

            if not explainability_bool:
                return classification_heads
            else:
                return classification_heads.cpu()


class MLPArchitecture(torch.nn.Module):
    """
    base architecture, used for finetuning the transformer model.
    """

    def __init__(
        self,
        n_tags: int,
        dropout_rate: float,
        device,
        transformer_output_length: int,
        embedding_statistics: torch.Tensor,
        trained_llm_model,
    ):
        super().__init__()
        self.device = device
        self.embedding_statistics = embedding_statistics.to(device)
        self.transformer_output_length = transformer_output_length
        # self.final_output_length = transformer_output_length * 4
        self.trained_llm_model = trained_llm_model
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.activation_function = nn.SELU()

        self.n_tags = n_tags
        midlayer_len = self.transformer_output_length // 8
        # second_midlayer_len = first_midlayer_len // 4

        self.extra_layer = torch.nn.Linear(4, 1)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            nn.Tanh(),
            torch.nn.LayerNorm(self.transformer_output_length),
            torch.nn.Linear(self.transformer_output_length, midlayer_len),
            torch.nn.Dropout(dropout_rate),
            # self.activation_function,
            # torch.nn.LayerNorm(first_midlayer_len),
            # torch.nn.Linear(first_midlayer_len, second_midlayer_len),
            # torch.nn.Dropout(dropout_rate),
            nn.SELU(),
            torch.nn.LayerNorm(midlayer_len),
            torch.nn.Linear(midlayer_len, 1),
        )

    def forward(self, inputs):
        with torch.no_grad():
            # size = (batch_size, self.transformer_output_length=384)
            llm_embeddings = self.trained_llm_model(inputs, return_embeddings=True)

        tags_differences = torch.zeros(
            (llm_embeddings.shape[0], self.n_tags, self.transformer_output_length, 3),
            device=self.device,
        )
        for tag_id in range(self.n_tags):
            # size = (3, self.transformer_output_length=384)
            embedding_stats_one_id = self.embedding_statistics[tag_id]
            # differences_one_tag = torch.cat(
            #     [llm_embeddings, differences_one_tag], axis=1
            # )

            # tags_differences[:, tag_id, :, 0] = llm_embeddings
            for i in range(3):
                tags_differences[:, tag_id, :, i] = (
                    llm_embeddings - embedding_stats_one_id[i]
                ) ** 2
            # tags_differences[:, tag_id, :, 2] = (
            #     llm_embeddings - embedding_stats_one_id[1]
            # ) ** 2
            # tags_differences[:, tag_id, :, 3] = (
            #     llm_embeddings - embedding_stats_one_id[2]
            # ) ** 2

        # size = (llm_embeddings.shape[0], self.n_tags, self.final_output_length)
        tags_differences = self.extra_layer(tags_differences)[..., 0]

        classification_heads = self.output_layer(tags_differences)[..., 0]
        return classification_heads


### TRAINING TRANSFORMERS ###


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
        self.dropout_rate = dropout_rate
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.model = TransformerArchitecture(
            model_name_or_path,
            self.ids_each_level,
            list(self.tagname_to_tagid.keys()),
            self.tokenizer,
            training_device,
            dropout_rate,
            output_length,
            n_freezed_layers=n_freezed_layers,
        )
        self.val_params = val_params

        self.training_device = training_device
        self.tags_proportions = tags_proportions

        self.loss = FocalLoss(
            tag_token_proportions=tags_proportions,
            gamma=loss_gamma,
            proportions_pow=proportions_pow,
            device=self.training_device,
        )

    def forward(self, inputs, return_embeddings: bool):
        output = self.model(inputs, return_embeddings)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch, return_embeddings=False)
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
        outputs = self(batch, return_embeddings=False)
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

        scheduler = StepLR(optimizer, step_size=1, gamma=0.4)

        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def get_loaders(self, dataset, params):
        set = ExcerptsDataset(
            dataset,
            self.tagname_to_tagid,
            self.tokenizer,
            self.max_len,
        )
        loader = DataLoader(set, **params, pin_memory=True)
        return loader


class TrainingMLP(pl.LightningModule):
    """
    pytorch lightning structure used for finetuning the trasformer.
    """

    def __init__(self, trained_llm_model, mlp_train_dataset):
        super().__init__()
        self.lr = 100 * trained_llm_model.lr
        self.weight_decay = trained_llm_model.weight_decay
        self.adam_epsilon = trained_llm_model.adam_epsilon
        self.transformer_output_length = trained_llm_model.transformer_output_length
        self.dropout_rate = trained_llm_model.dropout_rate
        self.tagname_to_tagid = trained_llm_model.tagname_to_tagid
        self.max_len = trained_llm_model.max_len
        self.training_device = trained_llm_model.training_device
        self.tokenizer = trained_llm_model.tokenizer
        self.trained_llm_model = trained_llm_model.model.to(self.training_device)
        self.val_params = trained_llm_model.val_params
        # self.val_params['batch_size'] = self.val_params['batch_size'] * 16

        self.tags_proportions = trained_llm_model.tags_proportions

        self.loss = trained_llm_model.loss
        self.embeddings_statistics = self._get_embedding_statistics(
            mlp_train_dataset
        ).to(self.training_device)

        self.mlp_architecture = MLPArchitecture(
            len(self.tagname_to_tagid),
            self.dropout_rate,
            self.training_device,
            self.transformer_output_length,
            self.embeddings_statistics,
            trained_llm_model=self.trained_llm_model,
        )

    def forward(self, inputs):
        output = self.mlp_architecture(inputs)
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

        # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def get_loaders(self, dataset, params):
        set = ExcerptsDataset(
            dataset,
            self.tagname_to_tagid,
            self.tokenizer,
            self.max_len,
        )
        loader = DataLoader(set, **params, pin_memory=True)
        return loader

    def _generate_embeddings(self, dataset):
        predictions_loader = self.get_loaders(
            dataset,
            self.val_params,
        )

        if torch.cuda.is_available():
            testing_device = "cuda"
        else:
            testing_device = "cpu"

        self.trained_llm_model.to(testing_device)

        llm_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                predictions_loader,
                total=len(predictions_loader.dataset) // predictions_loader.batch_size,
            ):
                embeddings_one_batch = self.trained_llm_model(
                    {
                        "ids": batch["ids"].to(testing_device),
                        "mask": batch["mask"].to(testing_device),
                    },
                    return_embeddings=True,
                )
                llm_embeddings.append(embeddings_one_batch)

        if len(llm_embeddings) > 0:
            llm_embeddings = torch.cat(llm_embeddings, dim=0)
        else:
            llm_embeddings = torch.tensor([])

        return llm_embeddings

    def _get_embedding_statistics(self, dataset):
        general_embeddings = self._generate_embeddings(dataset)

        embeddings_per_tag = torch.zeros(
            (len(self.tagname_to_tagid), 3, self.transformer_output_length),
            requires_grad=False,
        )
        for tagname, tag_id in self.tagname_to_tagid.items():
            tagname_in_excerpt_tags_mask = torch.tensor(
                [
                    1 if tagname in one_excerpt_tags else 0
                    for one_excerpt_tags in dataset.target
                ]
            )
            embeddings_one_tag = general_embeddings[tagname_in_excerpt_tags_mask]

            original_embedding_size = embeddings_one_tag.shape[0]
            sample_size = min(10_000, original_embedding_size)
            kept_indices = torch.tensor(
                random.sample(range(original_embedding_size), sample_size)
            )

            embeddings_one_tag = embeddings_one_tag[kept_indices]
            # stats_one_id = torch.zeros((3, embeddings_one_tag.shape[1]))
            embeddings_per_tag[tag_id, 0] = embeddings_one_tag.quantile(q=0.2, axis=0)
            embeddings_per_tag[tag_id, 1] = embeddings_one_tag.quantile(q=0.5, axis=0)
            embeddings_per_tag[tag_id, 2] = embeddings_one_tag.quantile(q=0.8, axis=0)
            # stats_one_id.append(embeddings_one_tag.mean(axis=0))

            # embeddings_per_tag[tag_id, ] = stats_one_id

        return embeddings_per_tag


class LoggedClassificationModel(torch.nn.Module):
    """
    Logged transformers structure, done for space memory optimization
    Only contains needed varibles and functions for inference
    """

    def __init__(self, trained_model) -> None:
        super().__init__()
        self.trained_architecture = trained_model.mlp_architecture
        self.tokenizer = trained_model.tokenizer
        self.tagname_to_tagid = trained_model.tagname_to_tagid
        self.max_len = trained_model.max_len
        self.transformer_output_length = trained_model.transformer_output_length
        self.val_params = trained_model.val_params
        self.val_params["num_workers"] = 0
        self.parent_tags = _get_parent_tags(list(self.tagname_to_tagid.keys()))
        self.tags_proportions = trained_model.tags_proportions

        # self.base_embeddings = self._generate_embeddings(train_dataset)

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
        hypertuning_threshold: bool = False,
    ):
        """
        1) get raw predictions
        2) postprocess them to output an output compatible with what we want in the inference
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

        logit_predictions = []
        y_true = []

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
                    }
                )
                logit_predictions.append(logits)

        logit_predictions = torch.cat(logit_predictions, dim=0)
        logit_predictions = torch.sigmoid(logit_predictions).cpu().numpy()

        if hypertuning_threshold:
            y_true = np.concatenate(y_true)
            return logit_predictions, y_true

        else:
            thresholds = np.array(list(self.optimal_thresholds.values()))
            final_predictions = logit_predictions / thresholds

            outputs = [
                {
                    tagname: final_predictions[i, tagid]
                    for tagname, tagid in self.tagname_to_tagid.items()
                }
                for i in range(logit_predictions.shape[0])
            ]

            return outputs

    def generate_test_predictions(
        self, entries: List[str], apply_postprocessing: bool = True
    ):
        model_outputs = self.custom_predict(
            entries,
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
