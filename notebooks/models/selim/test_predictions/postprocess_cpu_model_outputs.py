import boto3
import os
from ast import literal_eval
import pandas as pd
from typing import List, Dict
from collections import defaultdict

pillars_1d_tags = [
    "Covid-19",
    "Casualties",
    "Context",
    "Displacement",
    "Humanitarian Access",
    "Shock/Event",
    "Information And Communication",
]

pillars_2d_tags = [
    "At Risk",
    "Priority Interventions",
    "Capacities & Response",
    "Humanitarian Conditions",
    "Impact",
    "Priority Needs",
]

secondary_tags = [
    "age",
    "gender",
    "affected_groups",
    "specific_needs_groups",
    "severity",
]


def get_preds_entry(
    preds_column: Dict[str, float],
    return_at_least_one=True,
    ratio_nb=1,
    return_only_one=False,
):
    if return_only_one:
        preds_entry = [
            sub_tag
            for sub_tag, ratio in preds_column.items()
            if ratio == max(list(preds_column.values()))
        ]
    else:
        preds_entry = [
            sub_tag for sub_tag, ratio in preds_column.items() if ratio > ratio_nb
        ]
        if return_at_least_one and len(preds_entry) == 0:
            preds_entry = [
                sub_tag
                for sub_tag, ratio in preds_column.items()
                if ratio == max(list(preds_column.values()))
            ]
    return preds_entry


def get_predictions_all(
    ratios_entries: List[Dict[str, float]],
    pillars_2d=pillars_2d_tags,
    pillars_1d=pillars_1d_tags,
    ratio_nb: int = 1,
):

    predictions = defaultdict(list)
    for ratio_proba_threshold_one_entry in ratios_entries:
        returns_sectors = ratio_proba_threshold_one_entry["primary_tags"]["sectors"]

        subpillars_2d_tags = ratio_proba_threshold_one_entry["primary_tags"][
            "subpillars_2d"
        ]
        subpillars_1d_tags = ratio_proba_threshold_one_entry["primary_tags"][
            "subpillars_1d"
        ]

        ratios_sectors_subpillars_2d = list(returns_sectors.values()) + list(
            subpillars_2d_tags.values()
        )

        if any([item >= ratio_nb for item in ratios_sectors_subpillars_2d]):
            preds_2d = get_preds_entry(subpillars_2d_tags, True, ratio_nb)
            preds_sectors = get_preds_entry(returns_sectors, True, ratio_nb)

        else:
            preds_2d = []
            preds_sectors = []

        predictions["sectors"].append(preds_sectors)
        predictions["subpillars_2d"].append(preds_2d)

        preds_1d = get_preds_entry(subpillars_1d_tags, False, ratio_nb)
        predictions["subpillars_1d"].append(preds_1d)

        returns_sec_tags = ratio_proba_threshold_one_entry["secondary_tags"]

        for secondary_tag in [
            "age",
            "gender",
            "affected_groups",
            "specific_needs_groups",
        ]:
            preds_one_sec_tag = get_preds_entry(
                returns_sec_tags[secondary_tag], False, ratio_nb
            )

            predictions[secondary_tag].append(preds_one_sec_tag)

        severity_tags = returns_sec_tags["severity"]
        if any(["Humanitarian Conditions" in item for item in preds_2d]):
            preds_severity = get_preds_entry(severity_tags, True, ratio_nb, True)
        else:
            preds_severity = []
        predictions["severity"].append(preds_severity)

    return predictions


def flatten(t: List[List]) -> List:
    return [item for sublist in t for item in sublist]


def convert_current_dict_to_previous_one(
    ratios_one_entry: Dict[str, float]
) -> Dict[str, Dict[str, float]]:

    # prim tags
    primary_tags_results = {"sectors": {}, "subpillars_2d": {}, "subpillars_1d": {}}

    # sec tags
    secondary_tags_results = {
        "age": {},
        "gender": {},
        "affected_groups": {},
        "specific_needs_groups": {},
        "severity": {},
    }

    for tag, number in ratios_one_entry.items():
        tag_levels = tag.split("->")
        if "subpillars" == tag_levels[0]:
            assert tag_levels[1] in pillars_1d_tags or tag_levels[1] in pillars_2d_tags

            if tag_levels[1] in pillars_1d_tags:
                subpillar_name = "subpillars_1d"
            else:
                subpillar_name = "subpillars_2d"

            primary_tags_results[subpillar_name]["->".join(tag_levels[1:])] = number

        elif "secondary_tags" == tag_levels[0]:
            assert tag_levels[1] in secondary_tags

            secondary_tags_results[tag_levels[1]][tag_levels[2]] = number

        else:
            if "sectors" == tag_levels[1]:
                primary_tags_results["sectors"][tag_levels[2]] = number

    outputs = {
        "primary_tags": primary_tags_results,
        "secondary_tags": secondary_tags_results,
    }

    return outputs


def get_endpoint_probas(df: pd.DataFrame, endpoint_name: str):
    client = boto3.session.Session().client(
        "sagemaker-runtime", region_name="us-east-1"
    )

    all_outputs = []
    batch_size = 2
    for i in range(0, df.shape[0], batch_size):
        inputs = df.iloc[i : i + batch_size][["excerpt"]]
        inputs["return_type"] = "default_analyis"
        inputs["analyis_framework_id"] = "all"

        # predictions
        inputs["return_prediction_labels"] = True

        # kw for interpretability
        inputs["interpretability"] = False
        # minimum ratio between proba and threshold to perform interpretability
        inputs["ratio_interpreted_labels"] = 0.5
        inputs["attribution_type"] = "Layer DeepLift"

        # kw for embeddings
        inputs["output_backbone_embeddings"] = False
        inputs["pooling_type"] = "['cls', 'mean_pooling']"
        inputs[
            "finetuned_task"
        ] = "['first_level_tags', 'secondary_tags', 'subpillars']"
        inputs["embeddings_return_type"] = "array"

        backbone_inputs_json = inputs.to_json(orient="split")

        response = client.invoke_endpoint(
            EndpointName="tmp-hum",
            Body=backbone_inputs_json,
            ContentType="application/json; format=pandas-split",
        )
        output = response["Body"].read().decode("ascii")

        all_outputs.append(output)

    return all_outputs


endpoint_outputs = get_endpoint_probas(
    df=...[["excerpt"]].sample(n=1), endpoint_name="..."
)

eval_outputs = []
for batch in endpoint_outputs:
    eval_batch = literal_eval(batch)
    output_ratios = eval_batch["raw_predictions"]
    eval_outputs.append(output_ratios)
eval_outputs = flatten(eval_outputs)

thresholds = eval_batch["thresholds"]

thresholds_as_in_first_model = convert_current_dict_to_previous_one(thresholds)

outputs_as_in_first_model = [
    convert_current_dict_to_previous_one(one_entry_preds)
    for one_entry_preds in eval_outputs
]

final_predictions = get_predictions_all(outputs_as_in_first_model)
