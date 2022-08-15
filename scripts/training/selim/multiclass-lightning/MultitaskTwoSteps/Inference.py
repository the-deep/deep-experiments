import os
from turtle import forward

# setting tokenizers parallelism to false adds robustness when dploying the model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dill import needs to be kept for more robustness in multimodel serialization
import dill

dill.extend(True)

import sys

sys.path.append(".")

import mlflow


class ClassificationInference(mlflow.pyfunc.PythonModel):
    def __init__(self, models):
        self.models = models
        super().__init__()

    def load_context(self, context):
        pass

    def predict(self, context, inputs):

        input_sentences = inputs["excerpt"]
        return_type = inputs["return_type"].values[0]

        if return_type == "custom_postprocessing":
            processing_function = inputs["processing_function"].values[0]
            kwargs = inputs["kwargs"].values[0]
            return processing_function(input_sentences, self.models, **kwargs)

        else:
            af_id = inputs["analyis_framework_id"].values[0]

            if af_id in self.models.keys():
                final_id = af_id
            else:
                final_id = "all"

            output_backbone = self.models["backbone"].get_transformer_outputs(
                input_sentences
            )
            X_MLPs = {"X": output_backbone}

            predictions = self.models[final_id].custom_predict(X_MLPs, testing=True)

            outputs = {
                "raw_predictions": predictions,
                "thresholds": self.models[final_id].optimal_thresholds,
            }
            return outputs
