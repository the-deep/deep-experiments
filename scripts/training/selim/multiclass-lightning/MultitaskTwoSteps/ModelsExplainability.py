# This script is a custom adaptation from the https://github.com/cdpierse/transformers-interpret library, to adapt to our custom models !!!!
# We only keep the multilabel classification pipeline and do not keep the visuaization
# The method is also more static in this script, as it is intendended for a specific usage (models interpretability when getting predictions)

import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable
import re

import torch
import torch.nn as nn
from torch.nn.modules.sparse import Embedding
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
        sep_id: int,
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


class SequenceClassificationExplainer:
    """
    Explainer for explaining attributions for models of type
    `{MODEL_NAME}ForSequenceClassification` from the Transformers package.

    Calculates attribution for `text` using the given model
    and tokenizer.

    Attributions can be forced along the axis of a particular output index or class name.
    To do this provide either a valid `index` for the class label's output or if the outputs
    have provided labels you can pass a `class_name`.

    This explainer also allows for attributions with respect to a particlar embedding type.
    This can be selected by passing a `embedding_type`. The default value is `0` which
    is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
    If a model does not take position ids in its forward method (distilbert) a warning will
    occur and the default word_embeddings will be chosen instead.


    """

    def __init__(self, model):
        """
        Args:
            model (PreTrainedModel): Pretrained huggingface Sequence Classification model.
            tokenizer (PreTrainedTokenizer): Pretrained huggingface tokenizer
            attribution_type (str, optional): The attribution method to calculate on. Defaults to "lig".
            custom_labels (List[str], optional): Applies custom labels to label2id and id2label configs.
                                                 Labels must be same length as the base model configs' labels.
                                                 Labels and ids are applied index-wise. Defaults to None.

        Raises:
            AttributionTypeNotSupportedError:
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

        #
        self.accepts_position_ids = False
        self.accepts_token_type_ids = False
        self.position_embeddings = None
        self.token_type_embeddings = None

        self.attribution_type = "lig"

        self.label2id = self.model.tagname_to_tagid
        self.id2label = {id: label for label, id in self.label2id.items()}
        self.labels = list(self.label2id.keys())

        self.attributions: Union[None, LIGAttributions] = None
        self.input_ids: torch.Tensor = torch.Tensor()

        self._single_node_output = False

        self.internal_batch_size = None
        self.n_steps = 50

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

        if isinstance(text, list):
            raise NotImplementedError("Lists of text are not currently supported.")

        text_ids = self.encode(text)
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # if no special tokens were added
        if len(text_ids) == len(input_ids):
            ref_input_ids = [self.ref_token_id] * len(text_ids)
        else:
            ref_input_ids = (
                [self.cls_token_id]
                + [self.ref_token_id] * len(text_ids)
                + [self.sep_token_id]
            )

        return (
            torch.tensor([input_ids], device=self.device),
            torch.tensor([ref_input_ids], device=self.device),
            len(text_ids),
        )

    def _make_input_reference_token_type_pair(
        self, input_ids: torch.Tensor, sep_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two tensors indicating the corresponding token types for the `input_ids`
        and a corresponding all zero reference token type tensor.
        Args:
            input_ids (torch.Tensor): Tensor of text converted to `input_ids`
            sep_idx (int, optional):  Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor(
            [0 if i <= sep_idx else 1 for i in range(seq_len)], device=self.device
        ).expand_as(input_ids)
        ref_token_type_ids = torch.zeros_like(
            token_type_ids, device=self.device
        ).expand_as(input_ids)

        return (token_type_ids, ref_token_type_ids)

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

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(input_ids)

    def _clean_text(self, text: str) -> str:
        text = re.sub("([.,!?()])", r" \1 ", text)
        text = re.sub("\s{2,}", " ", text)
        return text

    def encode(self, text: str = None) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids: torch.Tensor) -> list:
        "Decode 'input_ids' to string using tokenizer"
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    @property
    def predicted_class_index(self) -> int:
        "Returns predicted class index (int) for model with last calculated `input_ids`"
        # we call this before _forward() so it has to be calculated twice
        preds = self.model(self.input_ids)[0]
        self.pred_class = torch.argmax(torch.softmax(preds, dim=0)[0])
        return torch.argmax(torch.softmax(preds, dim=1)[0]).cpu().detach().numpy()

    @property
    def predicted_class_name(self):
        "Returns predicted class name (str) for model with last calculated `input_ids`"
        try:
            index = self.predicted_class_index
            return self.id2label[int(index)]
        except Exception:
            return self.predicted_class_index

    @property
    def word_attributions(self) -> list:
        "Returns the word attributions for model and the text provided. Raises error if attributions not calculated."
        if self.attributions is not None:
            return self.attributions.word_attributions
        else:
            raise ValueError(
                "Attributions have not yet been calculated. Please call the explainer on text first."
            )

    def _forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        encoded = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )
        preds = self.model(encoded)

        # if it is a single output node
        if len(preds[0]) == 1:
            self._single_node_output = True
            self.pred_probs = torch.sigmoid(preds)[0][0]
            return torch.sigmoid(preds)[:, :]

        self.pred_probs = torch.softmax(preds, dim=1)[0][self.selected_index]
        return torch.softmax(preds, dim=1)[:, self.selected_index]

    def _calculate_attributions(self, embeddings: Embedding, index: int = None, class_name: str = None):  # type: ignore
        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.text)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        self.attention_mask = self._make_attention_mask(self.input_ids)

        if index is not None:
            self.selected_index = index
        elif class_name is not None:
            if class_name in self.label2id.keys():
                self.selected_index = int(self.label2id[class_name])
            else:
                s = f"'{class_name}' is not found in self.label2id keys."
                s += "Defaulting to predicted index instead."
                warnings.warn(s)
                self.selected_index = int(self.predicted_class_index)
        else:
            self.selected_index = int(self.predicted_class_index)

        reference_tokens = [
            token.replace("Ġ", "") for token in self.decode(self.input_ids)
        ]
        lig = LIGAttributions(
            self._forward,
            embeddings,
            reference_tokens,
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
            self.attention_mask,
            position_ids=self.position_ids,
            ref_position_ids=self.ref_position_ids,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
        )
        lig.summarize()
        self.attributions = lig

    def _run(
        self,
        text: str,
        index: int = None,
        class_name: str = None,
    ) -> list:  # type: ignore

        embeddings = self.word_embeddings

        self.text = self._clean_text(text)

        self._calculate_attributions(
            embeddings=embeddings, index=index, class_name=class_name
        )
        return self.word_attributions  # type: ignore

    def __call__(
        self,
        text: str,
        index: int = None,
        class_name: str = None,
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> list:
        """
        Calculates attribution for `text` using the model
        and tokenizer given in the constructor.

        Attributions can be forced along the axis of a particular output index or class name.
        To do this provide either a valid `index` for the class label's output or if the outputs
        have provided labels you can pass a `class_name`.

        This explainer also allows for attributions with respect to a particlar embedding type.
        This can be selected by passing a `embedding_type`. The default value is `0` which
        is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
        If a model does not take position ids in its forward method (distilbert) a warning will
        occur and the default word_embeddings will be chosen instead.

        Args:
            text (str): Text to provide attributions for.
            index (int, optional): Optional output index to provide attributions for. Defaults to None.
            class_name (str, optional): Optional output class name to provide attributions for. Defaults to None.
            embedding_type (int, optional): The embedding type word(0) or position(1) to calculate attributions for. Defaults to 0.
            internal_batch_size (int, optional): Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. If internal_batch_size is None, then all evaluations are
                processed in one batch.
            n_steps (int, optional): The number of steps used by the approximation
                method. Default: 50.
        Returns:
            list: List of tuples containing words and their associated attribution scores.
        """

        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size
        return self._run(text, index, class_name)


class MultiLabelClassificationExplainer(SequenceClassificationExplainer):
    """
    Explainer for independently explaining label attributions in a multi-label fashion
    for models of type `{MODEL_NAME}ForSequenceClassification` from the Transformers package.
    Every label is explained independently and the word attributions are a dictionary of labels
    mapping to the word attributions for that label. Even if the model itself is not multi-label
    by the resulting word attributions treat the labels as independent.

    Calculates attribution for `text` using the given model
    and tokenizer. Since this is a multi-label explainer, the attribution calculation time scales
    linearly with the number of labels.

    This explainer also allows for attributions with respect to a particlar embedding type.
    This can be selected by passing a `embedding_type`. The default value is `0` which
    is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
    If a model does not take position ids in its forward method (distilbert) a warning will
    occur and the default word_embeddings will be chosen instead.
    """

    def __init__(self, model):
        super().__init__(model)
        self.num_labels = len(self.labels)

    @property
    def word_attributions(self) -> dict:
        "Returns the word attributions for model and the text provided. Raises error if attributions not calculated."

        return dict(
            zip(
                self.labels,
                [attr.word_attributions for attr in self.attributions],
            )
        )

    def __call__(
        self,
        text: str,
        embedding_type: int = 0,
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> dict:
        """
        Calculates attributions for `text` using the model
        and tokenizer given in the constructor. Attributions are calculated for
        every label output in the model.

        This explainer also allows for attributions with respect to a particlar embedding type.
        This can be selected by passing a `embedding_type`. The default value is `0` which
        is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
        If a model does not take position ids in its forward method (distilbert) a warning will
        occur and the default word_embeddings will be chosen instead.

        Args:
            text (str): Text to provide attributions for.
            embedding_type (int, optional): The embedding type word(0) or position(1) to calculate attributions for. Defaults to 0.
            internal_batch_size (int, optional): Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. If internal_batch_size is None, then all evaluations are
                processed in one batch.
            n_steps (int, optional): The number of steps used by the approximation
                method. Default: 50.

        Returns:
            dict: A dictionary of label to list of attributions.
        """
        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size

        self.attributions = []
        self.pred_probs = []
        self.labels = list(self.label2id.keys())
        self.label_probs_dict = {}
        for i in range(self.num_labels):
            explainer = SequenceClassificationExplainer(self.model)
            explainer(text, i, embedding_type)

            self.attributions.append(explainer.attributions)
            self.input_ids = explainer.input_ids
            self.pred_probs.append(explainer.pred_probs)
            self.label_probs_dict[self.id2label[i]] = explainer.pred_probs

        return self.word_attributions
