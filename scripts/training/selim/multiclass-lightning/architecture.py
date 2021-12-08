from pooling import Pooling
import torch
from transformers import AutoModel
import torch.nn.functional as F
from utils import flatten


class Model(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        ids_each_level,
        dropout_rate=0.3,
        output_length=384,
        dim_hidden_layer: int = 256
    ):
        super().__init__()
        self.ids_each_level = ids_each_level
        self.l0 = AutoModel.from_pretrained(model_name_or_path)
        self.pool = Pooling(word_embedding_dimension=output_length, pooling_mode="mean")
        self.comon_hidden = torch.nn.Linear(output_length, 1024)
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.hidden = [torch.nn.Linear(1024, dim_hidden_layer) for _ in ids_each_level]
        self.hidden = torch.nn.ModuleList(self.hidden)
        self.output_layer = [
            torch.nn.Linear(dim_hidden_layer, len(id_one_level)) for id_one_level in ids_each_level
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
        )

        output = F.selu(output["sentence_embedding"])
        output = self.dropout(output)
        output = F.selu(self.comon_hidden(output))
        output = self.dropout(output)
        heads = []
        for i in range(len(self.ids_each_level)):
            output_tmp = F.selu(self.hidden[i](output))
            output_tmp = self.dropout(output_tmp)
            output_tmp = self.output_layer[i](output_tmp)
            heads.append(output_tmp)
        return heads
