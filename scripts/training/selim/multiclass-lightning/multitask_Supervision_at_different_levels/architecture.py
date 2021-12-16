from pooling import Pooling
import torch
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F
from utils import flatten
from typing import List


class Model(torch.nn.Module):
    def __init__(
            self,
            model_name_or_path: str,
            ids_each_level: List[List[int]],
            dropout_rate: float,
            output_length: int,
            dim_hidden_layer: int,
            hidden_layer_ids: List[int]
    ):
        super().__init__()
        self.ids_each_level = ids_each_level
        # hidden_layer_ids are explained in the function `get_first_level_ids` in the file `model.py`
        self.hidden_layer_ids = hidden_layer_ids
        self.l0 = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.hidden = [torch.nn.Linear(output_length, dim_hidden_layer) for _ in ids_each_level]
        self.hidden = torch.nn.ModuleList(self.hidden)
        # self.batch_norm_MLP = torch.nn.BatchNorm1d(dim_hidden_layer)
        # self.batch_norm_hidden_state = torch.nn.BatchNorm1d(output_length)

        self.output_layer = [
            torch.nn.Linear(dim_hidden_layer, len(id_one_level)) for id_one_level in ids_each_level
        ]
        self.output_layer = torch.nn.ModuleList(self.output_layer)

    def forward(self, inputs):
        output = self.l0(
            inputs["ids"],
            attention_mask=inputs["mask"],
            output_hidden_states=True
        )
        hidden_states = output.hidden_states
        heads = []
        for i in range(len(self.ids_each_level)):
            hidden_state_layer = F.softsign(hidden_states[self.hidden_layer_ids[i]])
            hidden_state_layer = self.dropout(hidden_state_layer)

            output_tmp = F.softsign(self.hidden[i](hidden_state_layer))
            output_tmp = self.dropout(output_tmp)
            output_tmp = self.output_layer[i](output_tmp)[:, 0, :]
            heads.append(output_tmp)
        return heads
