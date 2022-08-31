# This script is a custom adaptation from the https://github.com/cdpierse/transformers-interpret library, to adapt to our custom models.
# We only keep the multilabel classification pipeline and do not keep the visuaization
# The method is also more static in this script, as it is intendended for a specific usage (models interpretability when getting predictions)

import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable
import re

import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients


class Attributions:
    def __init__(self, custom_forward: Callable, embeddings: nn.Module, tokens: list):
        self.custom_forward = custom_forward
        self.embeddings = embeddings
        self.tokens = tokens


class LIGAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: nn.Module,
        tokens: list,
        input_ids: torch.Tensor,
        ref_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: Optional[Union[int, Tuple, torch.Tensor, List]] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ref_token_type_ids: Optional[torch.Tensor] = None,
        ref_position_ids: Optional[torch.Tensor] = None,
        internal_batch_size: Optional[int] = None,
        n_steps: int = 50,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.target = target
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.ref_position_ids = ref_position_ids
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps

        self.lig = LayerIntegratedGradients(self.custom_forward, self.embeddings)

        self._attributions, self.delta = self.lig.attribute(
            inputs=self.input_ids,
            baselines=self.ref_input_ids,
            target=self.target,
            return_convergence_delta=True,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
            additional_forward_args=(self.attention_mask),
        )

    @property
    def word_attributions(self) -> list:
        wa = []
        for i, (word, attribution) in enumerate(
            zip(self.tokens, self.attributions_sum)
        ):
            wa.append((word, float(attribution.cpu().data.numpy())))
        return wa

    def summarize(self, end_idx=None):
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0)
        self.attributions_sum = self.attributions_sum[:end_idx] / torch.norm(
            self.attributions_sum[:end_idx]
        )


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

    def __init__(self, model):
        """
        Args:
            model: Finetuned model on our tasks.

        """

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

    def _make_input_reference_pair(
        self, text: Union[List, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
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
            Tuple[torch.Tensor, torch.Tensor, int]
        """

        encoded = self.encode(text)
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

    def _make_input_reference_position_id_pair(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns tensors for positional encoding of tokens for input_ids and zeroed tensor for reference ids.

        Args:
            input_ids (torch.Tensor): inputs to create positional encoding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        ref_position_ids = torch.zeros(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return (position_ids, ref_position_ids)

    def _clean_text(self, text: str) -> str:
        text = re.sub("([.,!?()])", r" \1 ", text)
        text = re.sub("\s{2,}", " ", text)
        return text

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
        encoded = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )
        preds = self.model(encoded)

        return torch.sigmoid(preds)[:, self.tmp_selected_index]

    def _treat_one_entry(self, text: str):

        cleaned_text = self._clean_text(text)

        (input_ids, ref_input_ids, attention_mask) = self._make_input_reference_pair(
            cleaned_text
        )

        (
            position_ids,
            ref_position_ids,
        ) = self._make_input_reference_position_id_pair(input_ids)

        reference_tokens = [token.replace("Ġ", "") for token in self.decode(input_ids)]

        return {
            "input_ids": input_ids,
            "ref_input_ids": ref_input_ids,
            "position_ids": position_ids,
            "ref_position_ids": ref_position_ids,
            "attention_mask": attention_mask,
            "reference_tokens": reference_tokens,
        }

    def _get_attributions(self, index: int, entry_based_kwargs) -> list:
        """
        Calculates attribution for `text` using the model
        and tokenizer given in the constructor.

        Attributions can be forced along the axis of a particular output index or class name.
        To do this provide either a valid `index` for the class label's output or if the outputs
        have provided labels you can pass a `class_name`.

        Args:
            text (str): Text to provide attributions for.
            index (int, optional): Optional output index to provide attributions for. Defaults to None.
        Returns:
            list: List of tuples containing words and their associated attribution scores.
        """

        self.tmp_selected_index = index

        lig = LIGAttributions(
            self._forward,
            self.word_embeddings,
            entry_based_kwargs["reference_tokens"],
            entry_based_kwargs["input_ids"],
            entry_based_kwargs["ref_input_ids"],
            entry_based_kwargs["attention_mask"],
            position_ids=entry_based_kwargs["position_ids"],
            ref_position_ids=entry_based_kwargs["ref_position_ids"],
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
        )
        lig.summarize()
        return lig.word_attributions

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

        entry_based_kwargs = self._treat_one_entry(text)
        for one_id in labels_ids:
            attributions[self.id2label[one_id]] = self._get_attributions(
                one_id, entry_based_kwargs
            )

        return attributions
