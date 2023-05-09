from ast import literal_eval
from typing import List, Dict
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
from get_data import pull_data, _get_geolocation_table


def _flatten(t: List[List]) -> List:
    """flatten list of lists"""
    return [item for sublist in t for item in sublist]


def _custom_eval(x):
    if str(x) == "nan":
        return {}
    if str(x) == "[None]":
        return {}
    if type(x) is list:
        return x
    if type(x) is dict:
        return x
    else:
        return literal_eval(x)


def _item2list(item):
    if type(item) is list:
        return list(set(item))
    else:
        return [item]


mapping_widgets = [
    "matrix2dWidget",
    "multiselectWidget",
    "no_common_matrix2dWidget",
    "organigramWidget",
    "selectWidget",
    "raw",
    "scaleWidget",
    "no_common_multiselectWidget",
    "matrix1dWidget",
]

original_levels_cols = [
    "Original first level",
    "Original second level",
    "Original third level",
]
date_regex = re.compile(r"\d\d-\d\d-\d\d\d\d")


def _get_all_nlp_outputs(
    data: pd.DataFrame,
    hum_mapping_sheet: pd.DataFrame,
    geo_locations_dict: Dict[int, str],
):
    mapping_dict = defaultdict(
        list
    )  # {one_kw: [f"first_level_tags->Affected->{one_kw.capitalize()}"] for one_kw in ['migrants', 'affected', 'non displaced', 'displaced']}

    too_many_rows, no_mapping = set(), set()

    nlp_all_outputs = []

    for one_output in tqdm(data["original_tags"].tolist()):
        nlp_one_output = defaultdict(list)

        ####### dates
        dates_tmp = []

        for one_date_widget_type in ["dateRangeWidget", "dateWidget"]:
            if one_date_widget_type in one_output:
                dates_tmp.extend(
                    [
                        one_date
                        for one_date in one_output[one_date_widget_type]
                        if one_date is not None
                    ]
                )

        ##### geo_locations
        if "geoWidget" in one_output:
            geo_location_output = [
                geo_locations_dict.get(one_loc_id)
                for one_loc_id in one_output["geoWidget"]
            ]
        else:
            geo_location_output = []

        ##### nlp mapping widgets
        for one_widget_type in mapping_widgets:
            if one_widget_type in one_output:
                outputs_one_widget = one_output[one_widget_type]

                for item in outputs_one_widget:
                    if not str(item) in ["nan", "none", "", "n/a"]:
                        if item.isdigit():
                            geo_location_output.append(
                                geo_locations_dict.get(int(item))
                            )

                        elif date_regex.match(item):
                            dates_tmp.append(item)

                        else:
                            all_items = item.strip().split("->")

                            last_item = all_items[-1]

                            if item not in mapping_dict:
                                if len(all_items) == 1:
                                    # secondary tags or isolated items
                                    mapping_row = hum_mapping_sheet[
                                        hum_mapping_sheet.apply(
                                            lambda x: any(
                                                [
                                                    last_item == x[one_level_original]
                                                    for one_level_original in original_levels_cols
                                                ]
                                            ),
                                            axis=1,
                                        )
                                    ].copy()
                                # subpillars, subsectors
                                else:  # len(all_items) == 2:
                                    second_last_item = all_items[-2]
                                    mapping_row = hum_mapping_sheet[
                                        hum_mapping_sheet.apply(
                                            lambda x: second_last_item
                                            == x["Original first level"]
                                            and last_item == x["Original second level"],
                                            axis=1,
                                        )
                                    ].copy()

                                if len(mapping_row) == 1:
                                    one_mapped_item = mapping_row.iloc[0]["mapped_nlp"]

                                    nlp_one_output["nlp_tags"].extend(one_mapped_item)
                                    mapping_dict[item] = one_mapped_item

                                elif len(mapping_row) > 1:
                                    all_mapped_nlp = (
                                        mapping_row["mapped_nlp"].apply(str).tolist()
                                    )
                                    if len(set(all_mapped_nlp)) == 1:
                                        one_mapped_item = mapping_row.iloc[0][
                                            "mapped_nlp"
                                        ]

                                        nlp_one_output["nlp_tags"].extend(
                                            one_mapped_item
                                        )
                                        mapping_dict[item] = one_mapped_item
                                    else:
                                        first_level_mapped_row = hum_mapping_sheet[
                                            hum_mapping_sheet.apply(
                                                lambda x: x["Original first level"]
                                                == last_item
                                                and str(x["Original second level"])
                                                == "nan",
                                                axis=1,
                                            )
                                        ].copy()

                                        if len(first_level_mapped_row) == 1:
                                            one_mapped_item = (
                                                first_level_mapped_row.iloc[0][
                                                    "mapped_nlp"
                                                ]
                                            )

                                            nlp_one_output["nlp_tags"].extend(
                                                one_mapped_item
                                            )
                                            mapping_dict[item] = one_mapped_item
                                        else:
                                            too_many_rows.add(item)
                                            mapping_dict[item] = "too_many_rows"
                                else:
                                    no_mapping.add(item)
                                    mapping_dict[item] = "no_mapping"

                            elif mapping_dict[item] not in [
                                "no_mapping",
                                "too_many_rows",
                            ]:
                                nlp_one_output["nlp_tags"].extend(mapping_dict[item])

        nlp_one_output["geo_location"] = [
            one_loc for one_loc in geo_location_output if one_loc is not None
        ]

        if len(dates_tmp) > 0:
            dates_output = dates_tmp[0]
        else:
            dates_output = "-"

        nlp_one_output["excerpt_date"] = dates_output

        nlp_one_output["nlp_tags"] = list(set(nlp_one_output["nlp_tags"]))

        nlp_all_outputs.append(nlp_one_output)
    return nlp_all_outputs


def _pull_data_from_db(project_ids: List[int]):
    final_data = pd.DataFrame()

    for one_proj_id in tqdm(project_ids):
        final_data = pd.concat([pull_data(one_proj_id), final_data])

    return final_data


def get_final_classification_df(
    project_ids: List[int],
    mapping_sheet: pd.DataFrame,
):
    geolocation_df = _get_geolocation_table()
    geo_locations_dict = dict(
        zip(geolocation_df["id"].tolist(), geolocation_df["title"].tolist())
    )

    raw_classification_df = _pull_data_from_db(project_ids)
    data = raw_classification_df.copy().rename(columns={"outputs": "original_tags"})
    data["original_tags"] = data["original_tags"].apply(_custom_eval)
    data["raw_original_tags"] = data["original_tags"].apply(
        lambda x: list(set(_flatten([_item2list(item) for item in list(x.values())])))
    )

    mapping_sheet["mapped_nlp"] = mapping_sheet["mapped_nlp"].apply(_custom_eval)
    nlp_all_outputs = _get_all_nlp_outputs(data, mapping_sheet, geo_locations_dict)
    data["excerpt_date"] = [
        one_excerpt_tags["excerpt_date"] for one_excerpt_tags in nlp_all_outputs
    ]
    data["geo_location"] = [
        one_excerpt_tags["geo_location"] for one_excerpt_tags in nlp_all_outputs
    ]
    data["nlp_tags"] = [
        one_excerpt_tags["nlp_tags"] for one_excerpt_tags in nlp_all_outputs
    ]
    return data
