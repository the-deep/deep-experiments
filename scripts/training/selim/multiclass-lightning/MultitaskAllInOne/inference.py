import os

# setting tokenizers parallelism to false adds robustness when dploying the model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dill import needs to be kept for more robustness in multimodel serialization
import dill

dill.extend(True)

import sys

sys.path.append(".")

import mlflow


class TransformersPredictionsWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def load_context(self, context):
        pass

    def predict(self, context, inputs):

        input_sentences = inputs["excerpt"]
        return_type = inputs["return_type"].values[0]

        if return_type == "custom_postprocessing":
            processing_function = inputs["processing_function"].values[0]
            return processing_function(input_sentences, self.model, **kwargs)

        else:

            predictions = self.model.custom_predict(input_sentences, testing=True)

            outputs = {
                "raw_predictions": predictions,
                "thresholds": self.model.optimal_thresholds,
            }
            return outputs
