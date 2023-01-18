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
        sentences = inputs["lead_sentences"].tolist()
        forced_setup = (
            None
            if "forced_setup" not in inputs.columns
            else inputs["forced_setup"].iloc[0]
        )
        predictions = self.model.get_highlights(sentences, forced_setup)
        return predictions
