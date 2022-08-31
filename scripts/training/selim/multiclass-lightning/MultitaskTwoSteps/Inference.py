import torch
from ModelsExplainability import MultiLabelClassificationExplainer

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
        self.models = {
            model_name: model.to(torch.device("cpu"))
            for model_name, model in models.items()
        }
        super().__init__()

    def load_context(self, context):
        pass

    def predict(self, context, inputs):

        input_sentences = inputs["excerpt"].tolist()
        input_ids = inputs["entry_id"].tolist()
        n_entries = len(input_sentences)
        return_type = inputs["return_type"].values[0]
        interpretability_bool = inputs["return_type"].values[0]

        if return_type == "default_analyis":
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

            if interpretability_bool:
                prediction_specific_label_ids = inputs[
                    "prediction_specific_label_ids"
                ].tolist()
                interpretability_results = {}

                for i in range(n_entries):
                    cls_explainer = MultiLabelClassificationExplainer(
                        self.models["backbone"]
                    )
                    one_sentence = input_sentences[i]
                    one_entry_id = input_ids[i]
                    if prediction_specific_label_ids[0] is None:
                        prediction_one_entry = predictions[i]
                        final_predictions = [
                            label
                            for label, ratio in prediction_one_entry.items()
                            if ratio >= 1
                        ]
                        attributions_one_entry = cls_explainer(
                            one_sentence, final_predictions
                        )
                    else:
                        attributions_one_entry = cls_explainer(
                            one_sentence, prediction_specific_label_ids[i]
                        )
                    interpretability_results[one_entry_id] = attributions_one_entry
                return interpretability_results

            return outputs
        else:
            processing_function = inputs["processing_function"].values[0]
            kwargs = inputs["kwargs"].values[0]
            return processing_function(input_sentences, self.models, **kwargs)
