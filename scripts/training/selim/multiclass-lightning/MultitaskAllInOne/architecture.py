from pooling import Pooling
import torch
from transformers import AutoModel
from utils import flatten
import numpy as np
import torch.nn.functional as F
import copy


class Model(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path,
        ids_each_level,
        dropout_rate: float,
        output_length: int,
    ):
        super().__init__()
        self.ids_each_level = ids_each_level
        self.n_heads = len(flatten(self.ids_each_level))

        self.common_backbone = AutoModel.from_pretrained(
            model_name_or_path, output_hidden_states=True
        )
        self.common_backbone.encoder.layer = self.common_backbone.encoder.layer[:-1]

        """# freeze embeddings
        for param in self.common_backbone.embeddings.parameters():
            param.requires_grad = False
        # freeze two first layers
        for layer in self.common_backbone.encoder.layer[:2]:
            for param in layer.parameters():
                param.requires_grad = False"""

        self.pool = Pooling(
            word_embedding_dimension=output_length,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=True,
        )

        self.LayerNorm_specific_hidden = torch.nn.ModuleList(
            [torch.nn.LayerNorm(output_length * 2) for _ in range(self.n_heads)]
        )

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.specific_layer = torch.nn.ModuleList(
            [
                AutoModel.from_pretrained(model_name_or_path).encoder.layer[-1]
                for _ in range(self.n_heads)
            ]
        )

        self.output_layer = torch.nn.ModuleList(
            [
                torch.nn.Linear(output_length * 2, len(id_one_level))
                for id_one_level in flatten(ids_each_level)
            ]
        )

    def forward(self, inputs):

        common_output = self.common_backbone(
            inputs["ids"],
            attention_mask=inputs["mask"],
        ).last_hidden_state

        heads = [
            self.output_layer[tag_id](
                self.LayerNorm_specific_hidden[tag_id](
                    self.dropout(
                        self.pool(
                            {
                                "token_embeddings": self.specific_layer[tag_id](
                                    common_output
                                )[0],
                                "attention_mask": inputs["mask"],
                            }
                        )["sentence_embedding"]
                    )
                )
            )
            for tag_id in range(self.n_heads)
        ]

        return torch.cat(heads, dim=1)