import sys

sys.path.append(".")

import mlflow


class TransformersQAWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        predictions = self.model.custom_predict(model_input, testing=True)
        return predictions
