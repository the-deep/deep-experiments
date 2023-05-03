# This script is a custom adaptation from the https://github.com/cdpierse/transformers-interpret library, to adapt to our custom models.
# We only keep the multilabel classification pipeline and do not keep the visuaization.
# The method is also more static in this script, as it is intendended for a specific usage (models interpretability when getting predictions).
# We also integrate other explainability pipelines from captum.

import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable
import re

import torch
import torch.nn as nn
from captum.attr import (
    LayerIntegratedGradients,
    LayerDeepLift,
)

ATTRIBUTION_TYPES = [
    "Layer DeepLift",
    "Layer Integrated Gradients",
]


class OneLabelModel(nn.Module):
    def __init__(self, model: nn.Module, selected_index):
        super().__init__()
        self.model = model
        self.selected_index = selected_index

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        custom forward function
        """
        encoded = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )
        preds = self.model(encoded)

        return torch.sigmoid(preds)[:, self.selected_index]


class MultiLabelClassificationExplainer:
    """
    Explainer for independently explaining label attributions in a multi-label fashion
    for custom NLP models.
    Every label is explained independently and the word attributions are a dictionary of labels
    mapping to the word attributions for that label. Even if the model itself is not multi-label
    by the resulting word attributions treat the labels as independent.

    Calculates attribution for `text` using the given model
    and tokenizer. Since this is a multi-label explainer, the attribution calculation time scales
    linearly with the number of labels.
    """

    def __init__(self, model, attribution_type: str):
        """
        Args:
            model: Finetuned model on our tasks.

        """

        assert (
            attribution_type in ATTRIBUTION_TYPES
        ), f"'attribution_type' parameter must be in {ATTRIBUTION_TYPES}"
        self.attribution_type = attribution_type

        self.model = model

        self.tokenizer = model.tokenizer
        self.ref_token_id = self.tokenizer.pad_token_id

        self.sep_token_id = (
            self.tokenizer.sep_token_id
            if self.tokenizer.sep_token_id is not None
            else self.tokenizer.eos_token_id
        )
        self.cls_token_id = (
            self.tokenizer.cls_token_id
            if self.tokenizer.cls_token_id is not None
            else self.tokenizer.bos_token_id
        )

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.word_embeddings = (
            self.model.trained_architecture.common_backbone.get_input_embeddings()
        )

        self.label2id = self.model.tagname_to_tagid
        self.id2label = {id: label for label, id in self.label2id.items()}
        self.labels = list(self.label2id.keys())
        self.num_labels = len(self.labels)

        self.internal_batch_size = None
        self.n_steps = 50
        self._single_node_output = False

    def _clean_text(self, text: str) -> str:
        text = re.sub("([.,!?()])", r" \1 ", text)
        text = re.sub("\s{2,}", " ", text)
        return text

    def _make_input_reference_pair(
        self, text: Union[List, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenizes `text` to numerical token id  representation `input_ids`,
        as well as creating another reference tensor `ref_input_ids` of the same length
        that will be used as baseline for attributions. Additionally
        the length of text without special tokens appended is prepended is also
        returned.

        Args:
            text (str): Text for which we are creating both input ids
            and their corresponding reference ids

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """

        cleaned_text = self._clean_text(text)

        encoded = self.encode(cleaned_text)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        ref_input_ids = (
            [self.cls_token_id]
            + [self.ref_token_id] * (len(input_ids) - 2)
            + [self.sep_token_id]
        )

        return (
            torch.tensor([input_ids], device=self.device),
            torch.tensor([ref_input_ids], device=self.device),
            torch.tensor([attention_mask], device=self.device),
        )

    def encode(self, text: str) -> list:
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            return_attention_mask=True,
        )

    def decode(self, input_ids: torch.Tensor) -> list:
        "Decode 'input_ids' to string using tokenizer"
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        custom forward function
        """
        encoded = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )
        preds = self.model(encoded)

        return torch.sigmoid(preds)[:, self.tmp_selected_index]

    def _return_attributions_one_pass(self):
        """
        Different attribution types:inspired from https://captum.ai/api/attribution.html

        Returns:
            list: List of tuples containing words and their associated attribution scores.
        """

        if self.attribution_type == "Layer Integrated Gradients":
            lig = LayerIntegratedGradients(self._forward, self.word_embeddings)

            self._attributions = lig.attribute(
                inputs=self.input_ids,
                baselines=self.ref_input_ids,
                target=None,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
                additional_forward_args=(self.attention_mask),
            )

        elif self.attribution_type == "Layer DeepLift":

            one_label_model = OneLabelModel(self.model, self.tmp_selected_index)
            lig = LayerDeepLift(one_label_model, self.word_embeddings)

            self._attributions = lig.attribute(
                inputs=self.input_ids,
                baselines=self.ref_input_ids,
                target=None,
                additional_forward_args=(self.attention_mask),
            )

        else:
            raise ValueError("Something's wrong with the parameter 'attribution_type'")

        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0)
        self.attributions_sum = self.attributions_sum / torch.norm(
            self.attributions_sum
        )
        word_attributions = []
        for word, attribution in zip(self.reference_tokens, self.attributions_sum):
            word_attributions.append((word, float(attribution.cpu().data.numpy())))
        return word_attributions

    def __call__(
        self,
        text: str,
        treated_tags: List[str] = None,
    ) -> dict:
        """
        Calculates attributions for `text` using the model
        and tokenizer given in the constructor. Attributions are calculated for
        every label output in the model.

        Args:
            text (str): Text to provide attributions for.
            embedding_type (int, optional): The embedding type word(0) or position(1) to calculate attributions for. Defaults to 0.

        Returns:
            dict: A dictionary of label to list of attributions.
        """

        attributions = {}

        if treated_tags is None:
            labels_ids = [i for i in range(self.num_labels)]
        else:
            labels_ids = [self.label2id[label_id] for label_id in treated_tags]

        (
            self.input_ids,
            self.ref_input_ids,
            self.attention_mask,
        ) = self._make_input_reference_pair(text)

        self.reference_tokens = [
            token.replace("Ġ", "") for token in self.decode(self.input_ids)
        ]

        for one_id in labels_ids:
            self.tmp_selected_index = one_id
            attributions[self.id2label[one_id]] = self._return_attributions_one_pass()

        return attributions
