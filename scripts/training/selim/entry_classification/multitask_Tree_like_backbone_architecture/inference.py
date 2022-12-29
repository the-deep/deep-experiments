import os

# setting tokenizers parallelism to false adds robustness when dploying the model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dill import needs to be kept for more robustness in multimodel serialization
import dill

dill.extend(True)


import sys

sys.path.append(".")

import mlflow
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

        returned_ratios = []
        returned_preds = []
        input_sentences = inputs["sentences"]
        minimum_ratio = inputs["minimum_ratio"]

        for sentence in input_sentences:
            raw_predictions_tmp = {}
            final_predictions_tmp = {}

            raw_predictions_present_tags = self.models[
                "present_prim_tags"
            ].custom_predict(sentence, testing=True)
            raw_predictions_tmp["present_prim_tags"] = raw_predictions_present_tags

            if np.any(
                [
                    raw_predictions_present_tags[col] < minimum_ratio
                    for col in ["sectors", "subpillars_2d"]
                ]
            ):
                preds_subpillars_2d = []
                final_predictions_sectors = []
            else:
                raw_predictions_pillars_2d = self.models["pillars_2d"].custom_predict(
                    sentence, testing=True
                )
                final_predictions_pillars_2d = get_preds_entry(
                    raw_predictions_pillars_2d, False, minimum_ratio
                )
                raw_predictions_tmp["pillars_2d"] = raw_predictions_pillars_2d
                if len(final_predictions_pillars_2d) == 0:
                    final_predictions_sectors = []
                    final_predictions_subpillars_2d = []
                else:
                    preds_part_one_model = np.any(
                        [
                            pred in ["Humanitarian Conditions", "Impact"]
                            for pred in final_predictions_pillars_2d
                        ]
                    )
                    preds_part_two_model = np.any(
                        [
                            pred
                            in [
                                "Capacities & Response",
                                "Priority Interventions",
                                "At Risk",
                                "Priority Needs",
                            ]
                            for pred in final_predictions_pillars_2d
                        ]
                    )
                    if preds_part_one_model:
                        raw_predictions_subpillars_2d_part_one = self.models[
                            "subpillars_2d_part1"
                        ].custom_predict(sentence, testing=True)
                        final_predictions_subpillars_2d_part_one = get_preds_entry(
                            raw_predictions_subpillars_2d_part_one, False, minimum_ratio
                        )
                        raw_predictions_tmp[
                            "subpillars_2d_part1"
                        ] = raw_predictions_subpillars_2d_part_one

                    if preds_part_two_model:
                        raw_predictions_subpillars_2d_part_two = self.models[
                            "subpillars_2d_part2"
                        ].custom_predict(sentence, testing=True)
                        final_predictions_subpillars_2d_part_two = get_preds_entry(
                            raw_predictions_subpillars_2d_part_two, False, minimum_ratio
                        )
                        raw_predictions_tmp[
                            "subpillars_2d_part2"
                        ] = raw_predictions_subpillars_2d_part_two

                    final_predictions_subpillars_2d = (
                        final_predictions_subpillars_2d_part_one
                        + final_predictions_subpillars_2d_part_two
                    )

                    if len(final_predictions_subpillars_2d) == 0:
                        final_predictions_sectors = []
                    else:
                        raw_predictions_sectors = self.models["sectors"].custom_predict(
                            sentence, testing=True
                        )
                        final_predictions_sectors = get_preds_entry(
                            raw_predictions_sectors, False, minimum_ratio
                        )
                        raw_predictions_tmp["sectors"] = raw_predictions_sectors

            if raw_predictions_present_tags["subpillars_1d"] < minimum_ratio:
                final_predictions_subpillars_1d = []
            else:
                raw_predictions_pillars_1d = self.models["pillars_1d"].custom_predict(
                    sentence, testing=True
                )
                raw_predictions_tmp["pillars_1d"] = raw_predictions_pillars_1d
                final_predictions_pillars_1d = get_preds_entry(
                    raw_predictions_pillars_1d, False, minimum_ratio
                )

            if len(final_predictions_pillars_1d) == 0:
                final_predictions_subpillars_1d = []
            else:
                preds_part_one_model = np.any(
                    [
                        pred in ["Displacement", "Context"]
                        for pred in final_predictions_pillars_1d
                    ]
                )
                preds_part_two_model = np.any(
                    [
                        pred in ["Covid-19", "Shock/Event"]
                        for pred in final_predictions_pillars_1d
                    ]
                )
                preds_part_three_model = np.any(
                    [
                        pred
                        in [
                            "Casualties",
                            "Information And Communication",
                            "Humanitarian Access",
                        ]
                        for pred in final_predictions_pillars_1d
                    ]
                )
                if preds_part_one_model:
                    raw_predictions_subpillars_1d_part_one = self.models[
                        "subpillars_1d_part1"
                    ].custom_predict(sentence, testing=True)
                    final_predictions_subpillars_1d_part_one = get_preds_entry(
                        raw_predictions_subpillars_1d_part_one, False, minimum_ratio
                    )
                    raw_predictions_tmp[
                        "subpillars_1d_part1"
                    ] = raw_predictions_subpillars_1d_part_one

                if preds_part_two_model:
                    raw_predictions_subpillars_1d_part_two = self.models[
                        "subpillars_1d_part2"
                    ].custom_predict(sentence, testing=True)
                    final_predictions_subpillars_1d_part_two = get_preds_entry(
                        raw_predictions_subpillars_1d_part_two, False, minimum_ratio
                    )
                    raw_predictions_tmp[
                        "subpillars_1d_part2"
                    ] = raw_predictions_subpillars_1d_part_two

                if preds_part_three_model:
                    raw_predictions_subpillars_1d_part_three = self.models[
                        "subpillars_1d_part3"
                    ].custom_predict(sentence, testing=True)
                    final_predictions_subpillars_1d_part_three = get_preds_entry(
                        raw_predictions_subpillars_1d_part_three, False, minimum_ratio
                    )
                    raw_predictions_tmp[
                        "subpillars_1d_part3"
                    ] = raw_predictions_subpillars_1d_part_three

                final_predictions_subpillars_1d = (
                    final_predictions_subpillars_1d_part_one
                    + final_predictions_subpillars_1d_part_two
                    + final_predictions_subpillars_1d_part_three
                )

            # get predictions and append them
            final_predictions_tmp["sectors"] = final_predictions_sectors
            final_predictions_tmp["subpillars_2d"] = preds_subpillars_2d
            final_predictions_tmp["subpillars_1d"] = final_predictions_subpillars_1d
            returned_preds.append(final_predictions_tmp)
            returned_ratios.append(raw_predictions_tmp)

        outputs = {
            "raw_ratios": returned_ratios,
            "final_predictions": returned_preds,
            "thresholds": self.thresholds,
        }
        """raw_predictions = {}
        for tag_name, trained_model in self.models.items():

            predictions_one_model = trained_model.custom_predict(
                inputs, testing=True
            )
            raw_predictions[tag_name] = predictions_one_model
        """

        return outputs
