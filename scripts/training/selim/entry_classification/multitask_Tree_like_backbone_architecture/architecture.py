from pooling import Pooling
import torch
from transformers import AutoModel
import torch.nn.functional as F
from utils import flatten


class InitialModel(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        ids_each_level,
        dropout_rate: float,
        output_length: int,
        dim_hidden_layer: int,
        max_len: int,
    ):
        super().__init__()
        self.ids_each_level = ids_each_level
        self.l0 = AutoModel.from_pretrained(model_name_or_path)

        self.batch_norm_backbone = torch.nn.BatchNorm1d(max_len)
        self.batch_norm_specific_hidden = torch.nn.BatchNorm1d(max_len)
        self.batch_norm_common_hidden = torch.nn.BatchNorm1d(max_len)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.common_hidden_layer = torch.nn.Linear(output_length, dim_hidden_layer)

        self.specific_hidden_layer = [
            torch.nn.Linear(dim_hidden_layer, dim_hidden_layer // 2)
            for _ in ids_each_level
        ]
        self.specific_hidden_layer = torch.nn.ModuleList(self.specific_hidden_layer)

        self.output_layer = [
            torch.nn.Linear(dim_hidden_layer // 2, len(id_one_level))
            for id_one_level in ids_each_level
        ]
        self.output_layer = torch.nn.ModuleList(self.output_layer)

    def forward(self, inputs):
        backbone = self.l0(
            inputs["ids"],
            attention_mask=inputs["mask"],
        ).last_hidden_state

        # after backbone
        output = F.selu(backbone)
        output = self.dropout(output)
        output = self.batch_norm_backbone(output)

        # common hidden layer
        output = F.selu(self.common_hidden_layer(output))
        output = self.dropout(output)
        output = self.batch_norm_common_hidden(output)

        heads = []
        for i in range(len(self.ids_each_level)):
            # specific hidden layer
            output_tmp = F.leaky_relu(self.specific_hidden_layer[i](output))
            output_tmp = self.dropout(output_tmp)
            output_tmp = self.batch_norm_specific_hidden(output_tmp)

            # output layer
            output_tmp = self.output_layer[i](output_tmp)
            heads.append(output_tmp[:, 0, :])
        return heads


class NextModels(torch.nn.Module):
    def __init__(
        self,
        backbone,
        ids_each_level,
        dropout_rate: float,
        dim_hidden_layer: int,
    ):
        super().__init__()

        new_specific_hidden_layer = [
            torch.nn.Linear(dim_hidden_layer, dim_hidden_layer // 2)
            for _ in ids_each_level
        ]
        new_specific_hidden_layer = torch.nn.ModuleList(
            new_specific_hidden_layer
        )

        new_output_layer = [
            torch.nn.Linear(dim_hidden_layer // 2, len(id_one_level))
            for id_one_level in ids_each_level
        ]
        new_output_layer = torch.nn.ModuleList(new_output_layer)

        backbone.specific_hidden_layer = new_specific_hidden_layer
        backbone.output_layer = new_output_layer
        backbone.dropout_rate = dropout_rate
        self.backbone = backbone

    def forward(self, inputs):
        return self.backbone(inputs)
