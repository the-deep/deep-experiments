import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Dict
from transformers import AdamW, AutoTokenizer, AutoModel

from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import (
    flatten,
    get_tag_id_to_layer_id,
    get_first_level_ids,
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
        self.n_heads = len(flatten(self.ids_each_level))

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
                for id_one_level in flatten(ids_each_level)
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

    def get_loaders(self, dataset, params):

        set = CustomDataset(
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

    def forward(self, inputs):
        output = self.trained_architecture(inputs)
        return output

    def get_loaders(self, dataset, params):

        set = CustomDataset(
            dataset, self.tagname_to_tagid, self.tokenizer, self.max_len
        )
        loader = DataLoader(set, **params, pin_memory=True)
        return loader

    def get_transformer_outputs(self, dataset):
        """
        1) get raw predictions
        2) postprocess them to output an output compatible with what we want in the inference
        """

        test_loader = self.get_loaders(
            dataset,
            self.val_params,
        )

        if torch.cuda.is_available():
            testing_device = "cuda"
        else:
            testing_device = "cpu"

        self.to(testing_device)

        with torch.no_grad():

            backbone_outputs = [
                self(
                    {
                        "ids": batch["ids"].to(testing_device),
                        "mask": batch["mask"].to(testing_device),
                        "return_transformer_only": True,
                    }
                ).cpu()
                for batch in tqdm(
                    test_loader,
                    total=len(test_loader.dataset) // test_loader.batch_size,
                )
            ]

        backbone_outputs = torch.cat(backbone_outputs, dim=0)

        return backbone_outputs


# TODO: clean all.


class CustomDataset(Dataset):
    """
    transformers custom dataset
    """

    def __init__(self, dataframe, tagname_to_tagid, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.data = dataframe

        self.targets = None
        self.entry_ids = None

        if dataframe is None:
            self.excerpt_text = None

        elif type(dataframe) is str:
            self.excerpt_text = [dataframe]

        elif type(dataframe) is list:
            self.excerpt_text = dataframe

        elif type(dataframe) is pd.Series:
            self.excerpt_text = dataframe.tolist()

        else:
            self.excerpt_text = dataframe["excerpt"].tolist()
            df_cols = dataframe.columns
            if "target" in df_cols and "entry_id" in df_cols:
                self.targets = list(dataframe["target"])
                # self.entry_ids = list(dataframe["entry_id"])

        self.tagname_to_tagid = tagname_to_tagid
        self.tagid_to_tagname = list(tagname_to_tagid.keys())
        self.max_len = max_len

    def _delete_context(text: str) -> str:
        if text[0] == "[":
            n_strings = len(text)
            hook_end = 0
            while text[hook_end] != "]" and hook_end < 50:
                hook_end += 1
                if hook_end == (n_strings - 1):
                    return text

            if hook_end < 70:
                return text[(hook_end + 1) :].lstrip()
            else:
                return text.lstrip()

        else:
            return text.lstrip()

    def _clean_lgbt_words(text: str) -> str:
        def lgbt_in_word(word):
            return "lgbt" in word.lower() or "lgtb" in word.lower()

        def get_one_lgbt_token(word):
            if word[-1] in punctuation and word[-1] != "+":
                final_punct = word[-1]
                word = word[:-1]
            else:
                final_punct = ""

            return ("lgbt" + final_punct).rstrip()

        clean_text = copy(text)
        if lgbt_in_word(clean_text):
            clean_text = clean_text.replace(" +", "+")
            words = clean_text.split(" ")
            output_text = " ".join(
                [
                    one_word
                    if not lgbt_in_word(one_word)
                    else get_one_lgbt_token(one_word)
                    for one_word in words
                ]
            )
            return output_text

        else:
            return text

    def _preprocess_text(text: str) -> str:
        clean_text = copy(text)
        clean_text = delete_context(clean_text)
        # clean_text = delete_punctuation(clean_text)
        clean_text = clean_lgbt_words(clean_text)
        # clean_text = preprocess_one_sentence(clean_text)
        return clean_text

    def encode_example(self, excerpt_text: str, index=None, as_batch: bool = False):

        inputs = self.tokenizer(
            excerpt_text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        encoded = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
        }

        targets = None
        if self.targets:
            target_indices = [
                self.tagname_to_tagid[target]
                for target in self.targets[index]
                if target in self.tagname_to_tagid
            ]
            targets = np.zeros(len(self.tagname_to_tagid), dtype=int)
            targets[target_indices] = 1

            encoded["targets"] = (
                torch.tensor(targets, dtype=float) if targets is not None else None
            )

        return encoded

    def __len__(self):
        return len(self.excerpt_text)

    def __getitem__(self, index):
        excerpt_text = str(self.excerpt_text[index])
        return self.encode_example(excerpt_text, index)
