from pooling import Pooling
import torch
from transformers import AutoModel
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        dropout_rate: float = 0.3,
        output_length: int = 384,
        dim_hidden_layer: int = 256,
    ):
        super().__init__()

        self.l0 = AutoModel.from_pretrained(model_name_or_path)
        self.pool = Pooling(word_embedding_dimension=output_length, pooling_mode="cls")

        self.Norm_backbone = torch.nn.LayerNorm(output_length)
        self.Norm_specific_hidden = torch.nn.BatchNorm1d(dim_hidden_layer)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.specific_hidden_layer = torch.nn.Linear(output_length, dim_hidden_layer)
        self.output_layer = torch.nn.Linear(dim_hidden_layer, num_labels)

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

        output = torch.tanh(output["sentence_embedding"])
        output = self.dropout(output)
        output = self.Norm_backbone(output)

        output = F.selu(self.specific_hidden_layer(output))
        output = self.dropout(output)
        output = self.Norm_specific_hidden(output)

        # output layer
        output = self.output_layer(output)

        return output
