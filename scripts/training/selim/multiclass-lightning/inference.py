import os
#setting tokenizers parallelism to false adds robustness when dploying the model
#os.environ["TOKENIZERS_PARALLELISM"] = "false" 

#dill import needs to be kept for more robustness in multimodel serialization
import dill
dill.extend(True)


import sys
sys.path.append(".")

import mlflow


from get_outputs_user import get_predictions

class TransformersPredictionsWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.thresholds = {}

    def load_context(self, context):
        pass

    def add_model(self, model, model_name:str):
        self.models[model_name] = model
        self.thresholds[model_name] = model.optimal_thresholds

    def predict(self, context, model_input):

        raw_predictions = {}
        for tag_name, trained_model in self.models.items():

            predictions_one_model = trained_model.custom_predict(model_input, testing=True)
            raw_predictions[tag_name] = predictions_one_model

        post_processed_results = get_predictions(raw_predictions, self.thresholds)
        return raw_predictions, self.thresholds, post_processed_results
