from pooling import Pooling
import torch
from transformers import AutoModel
import torch.nn.functional as F
import copy


class Model(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        ids_each_level,
        dropout_rate=0.3,
        output_length=384,
        dim_hidden_layer: int = 256,
    ):
        super().__init__()
        self.ids_each_level = ids_each_level
        self.l0 = AutoModel.from_pretrained(model_name_or_path)
        self.pool = Pooling(word_embedding_dimension=output_length, pooling_mode="cls")

        self.LayerNorm_backbone = torch.nn.LayerNorm(output_length)
        self.LayerNorm_specific_hidden = torch.nn.LayerNorm(dim_hidden_layer)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.specific_hidden_layer = [
            torch.nn.Linear(output_length, dim_hidden_layer) for _ in ids_each_level
        ]
        self.specific_hidden_layer = torch.nn.ModuleList(self.specific_hidden_layer)

        self.output_layer = [
            torch.nn.Linear(dim_hidden_layer, len(id_one_level))
            for id_one_level in ids_each_level
        ]
        self.output_layer = torch.nn.ModuleList(self.output_layer)

    def forward(self, inputs):
        output = self.l0(
            inputs["ids"],
            attention_mask=inputs["mask"],
        )
        output = self.pool(
            {
                "token_embeddings": output.last_hidden_state,
                "attention_mask": inputs["mask"],
            }
        )["sentence_embedding"]

        last_hidden_states = [output.clone() for _ in self.ids_each_level]
        #last_hidden_states = torch.nn.ModuleList(last_hidden_states)

        heads = []
        for i in range(len(self.ids_each_level)):
            # specific hidden layer
            output_tmp = F.selu(last_hidden_states[i])
            output_tmp = self.dropout(output_tmp)
            output_tmp = self.LayerNorm_specific_hidden(output_tmp)

            # output layer
            output_tmp = self.output_layer[i](output_tmp)
            heads.append(output_tmp)

        return heads
