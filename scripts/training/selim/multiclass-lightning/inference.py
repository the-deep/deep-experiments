import sys

sys.path.append(".")

import mlflow
from model import Transformer
import torch

#dill import needs to be kept for more robustness in multimodel serialization
import dill
dill.extend(True)

class TransformersPredictionsWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        super().__init__()
        self.models = {}

    def load_context(self, context):
        pass

    def add_model(self, model, model_name:str):
        self.models[model_name] = model

    def predict(self, context, model_input):

        final_predictions = {}
        for tag_name, trained_model in self.models.items():

            predictions_one_model = trained_model.custom_predict(model_input, testing=True)
            final_predictions[tag_name] = predictions_one_model

        return final_predictions

class PythonPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.models = {}

    def load_context(self, context):
        pass

    def add_model(self, model, model_name:str):
        self.models[model_name] = model

    def predict(self, context, model_input):

        final_predictions = {}
        for tag_name, trained_model in self.models.items():

            predictions_one_model = trained_model.custom_predict(model_input, testing=True)
            final_predictions[tag_name] = predictions_one_model

        return final_predictions