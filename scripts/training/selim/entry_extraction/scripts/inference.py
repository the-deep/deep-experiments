import sys

sys.path.append(".")

import pandas as pd
import mlflow


class EntryExtractionWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model.eval()
        super().__init__()

    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        predictions = self.model.get_highlights(model_input)
        return pd.Series(predictions)
