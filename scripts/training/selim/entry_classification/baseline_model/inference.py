import os

# setting tokenizers parallelism to false adds robustness when dploying the model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dill import needs to be kept for more robustness in multimodel serialization
import dill

dill.extend(True)


import sys

sys.path.append(".")

import mlflow
import random
import numpy as np
from utils import get_preds_entry


class TransformersPredictionsWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.thresholds = {}

    def load_context(self, context):
        pass

    def add_model(self, model, model_name: str):
        self.models[model_name] = model
        self.thresholds[model_name] = model.optimal_thresholds

    def predict(self, context, inputs):

        input_sentences = inputs["excerpt"]
        return_type = inputs["return_type"].values[0]

        if return_type == "one_model":

            specific_model = inputs["model"].values[0]
            raw_predictions_one_model = self.models[specific_model].custom_predict(
                input_sentences, testing=True
            )
            outputs = {
                "raw_predictions": raw_predictions_one_model,
                "thresholds": self.thresholds[specific_model],
            }
            return outputs

        elif return_type == "custom_postprocessing":
            processing_function = inputs["processing_function"].values[0]
            minimum_ratio = inputs["processing_function"].values[0]
            return_at_least_one = inputs["processing_function"].values[0]
            output_columns = inputs["output_columns"].values[0]
            prim_tags_bool = inputs["prim_tags_bool"].values[0]
            return processing_function(
                input_sentences, 
                minimum_ratio, 
                return_at_least_one,
                output_columns,
                prim_tags_bool
                )

        elif return_type == "test":
            all_raw_preds = []
            for sentence in input_sentences:
                n_models = inputs["model"].values[0]
                test_models = ["present_prim_tags", "sectors", "pillars_2d"]
                if n_models > 3:
                    other_models = [
                        "pillars_1d",
                        "Impact",
                        "Context",
                        "Covid-19",
                        "Displacement",
                        "Humanitarian Access",
                        "Shock/Event",
                    ]
                    n_added_models = n_models - len(test_models)
                    added_models = random.sample(other_models, n_added_models)
                    test_models = test_models + added_models

                raw_predictions = {}
                for one_model in test_models:

                    predictions_one_model = self.models[one_model].custom_predict(
                        sentence, testing=True
                    )
                    raw_predictions[one_model] = predictions_one_model
            all_raw_preds.append(raw_predictions)

            outputs = {
                "raw_predictions": all_raw_preds,
                "thresholds": self.thresholds,
            }
            return outputs

        else:
            raw_predictions = {}
            for tag_name, trained_model in self.models.items():

                predictions_one_model = trained_model.custom_predict(
                    input_sentences, testing=True
                )
                raw_predictions[tag_name] = predictions_one_model
            outputs = {
                "raw_predictions": raw_predictions,
                "thresholds": self.thresholds,
            }
            return outputs
