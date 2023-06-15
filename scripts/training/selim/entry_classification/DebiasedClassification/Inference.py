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
from ast import literal_eval
from utils import _flatten


class ClassificationInference(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model.to(torch.device("cpu"))
        super().__init__()

    def load_context(self, context):
        pass

    def predict(self, context, inputs):

        input_sentences = inputs["excerpt"].tolist()
        n_entries = len(input_sentences)
        return_type = inputs["return_type"].values[0]

        if return_type == "default_analyis":

            output_backbone_embeddings_bool = inputs[
                "output_backbone_embeddings"
            ].values[0]
            return_prediction_labels_bool = inputs["return_prediction_labels"].values[0]
            interpretability_bool = inputs["interpretability"].values[0]

            outputs = {}

            if output_backbone_embeddings_bool:
                outputs_backbone = self.model.custom_predict(
                    input_sentences, return_transformer_only=True
                )
                # parameters retrieved from model
                transformer_output_length = self.model.transformer_output_length
                tot_labels_names = self.model.tagname_to_tagid.keys()
                possible_task_names = list(
                    set([item.split("->")[0] for item in tot_labels_names])
                )
                possible_pooling_types = ["cls", "mean_pooling"]
                n_possible_pooling_types = len(possible_pooling_types)

                # retured embedding is of the form [task1_cls_pooling, task1_mean_pooling, task2_cls_pooling, task2_mean_pooling ....]

                # parameters retried from call
                embedding_pooling_type = literal_eval(inputs["pooling_type"].values[0])
                embedding_finetuned_task = literal_eval(
                    inputs["finetuned_task"].values[0]
                )

                # sanity check for pooling types
                for one_pooling_type in embedding_pooling_type:
                    assert (
                        one_pooling_type in possible_pooling_types
                    ), f"'embedding_pooling_type' must be in {possible_pooling_types}"

                # sanity check for finetuned tasks
                for one_task in embedding_finetuned_task:
                    assert (
                        one_task in possible_task_names
                    ), f"'embedding_finetuned_task' must be in {possible_task_names}"

                # id of chosen pooling types in pooling possiblities
                to_be_returned_pooling_ids = [
                    i
                    for i, type in enumerate(possible_pooling_types)
                    if type in embedding_pooling_type
                ]

                # id of chosen finetuned tasks in pooling possiblities
                to_be_returned_task_ids = [
                    i
                    for i, type in enumerate(possible_task_names)
                    if type in embedding_finetuned_task
                ]

                # well locate embedding ids
                to_be_returned_ids = [
                    i * n_possible_pooling_types + j
                    for i in to_be_returned_pooling_ids
                    for j in to_be_returned_task_ids
                ]
                to_be_returned_embedding_ids = _flatten(
                    [
                        [id_tmp for id_tmp in range(id, id + transformer_output_length)]
                        for id in to_be_returned_ids
                    ]
                )

                try:

                    returned_embedding = outputs_backbone[
                        :, to_be_returned_embedding_ids
                    ]
                except Exception:
                    print("ids", to_be_returned_ids)
                    print("embedding_ids", to_be_returned_embedding_ids)

                embeddings_return_type = inputs["embeddings_return_type"].values[0]
                possible_returning_types = ["list", "array"]
                assert (
                    embeddings_return_type in possible_returning_types
                ), f"'embeddings_return_type' parameter must be in {possible_returning_types}"

                if embeddings_return_type == "list":
                    returned_embedding = returned_embedding.tolist()
                else:
                    returned_embedding = returned_embedding.numpy()

                outputs["output_backbone"] = returned_embedding

            if return_prediction_labels_bool or interpretability_bool:
                predictions = self.model.custom_predict(
                    input_sentences,
                    hypertuning_threshold=False,
                    return_transformer_only=False,
                )

            if return_prediction_labels_bool:
                outputs["raw_predictions"] = predictions
                outputs["thresholds"] = self.model.optimal_thresholds

            if interpretability_bool:
                ratio_interpreted_labels = inputs["ratio_interpreted_labels"].values[0]
                attribution_type = inputs["attribution_type"].values[0]

                ATTRIBUTION_TYPES = [
                    "Layer DeepLift",
                    "Layer Integrated Gradients",
                ]
                assert (
                    attribution_type in ATTRIBUTION_TYPES
                ), f"'attribution_type' parameter must be in {ATTRIBUTION_TYPES}"

                interpretability_results = []
                cls_explainer = MultiLabelClassificationExplainer(
                    self.model, attribution_type
                )

                for i in range(n_entries):
                    predictions_one_sentence = [
                        label
                        for label, ratio in predictions[i].items()
                        if ratio > ratio_interpreted_labels
                    ]
                    one_sentence = input_sentences[i]
                    if len(predictions_one_sentence) > 0:
                        attributions_one_entry = cls_explainer(
                            one_sentence, predictions_one_sentence
                        )
                    else:
                        attributions_one_entry = {}

                    interpretability_results.append(attributions_one_entry)

                outputs["interpretability_results"] = interpretability_results

            return outputs

        else:
            processing_function = inputs["processing_function"].values[0]
            kwargs = inputs["kwargs"].values[0]
            return processing_function(input_sentences, self.model, **kwargs)
