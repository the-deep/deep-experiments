import sys

sys.path.append(".")

import pandas as pd
import torch
import mlflow
from torch.utils.data import DataLoader

from data import SectorsDataset


class TransformersQAWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model.eval()
        super().__init__()

    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        dataset = SectorsDataset(model_input, self.tokenizer)
        val_params = {"batch_size": 16, "shuffle": False, "num_workers": 0}
        dataloader = DataLoader(dataset, **val_params)
        with torch.no_grad():
            predictions = [self.model.forward(batch) for batch in dataloader]
        predictions = torch.cat(predictions)
        predictions = predictions.argmax(1)
        return pd.Series(predictions)
