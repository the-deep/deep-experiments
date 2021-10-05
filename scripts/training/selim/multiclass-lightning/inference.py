import sys

sys.path.append(".")

import mlflow
from model import Transformer
import torch
import dill

class TransformersPredictionsWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        super().__init__()
        self.models = {}

    def load_context(self, context):
        pass

    def add_model(self, model:Transformer, model_name:str):
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

    def add_model(self, model:Transformer, model_name:str):
        self.models[model_name] = model

    def predict(self, context, model_input):

        models_list = list(self.models.keys())

        final_predictions = {}
        for i in range (len (models_list)):

            predictions_one_model = models_list[i].custom_predict(model_input, testing=True)
            final_predictions.update(predictions_one_model)  

        return final_predictions