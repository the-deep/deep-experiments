import sys

sys.path.append(".")

import pandas as pd
import mlflow


class EntryExtractionWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def load_context(self, context):
        pass

    def predict(self, context, inputs):
        sentences = inputs["return_type"].tolist()
        predictions = self.model.get_highlights(sentences)
        return predictions
