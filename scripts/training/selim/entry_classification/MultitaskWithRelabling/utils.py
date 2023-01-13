from ast import literal_eval
import random

random.seed(30)

import numpy as np
import pandas as pd
import re
from copy import copy
import warnings
from typing import Dict, List, Union, Tuple
from collections import Counter, defaultdict
import torch
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# GENERAL UTIL FUNCTIONS

languages = ["en", "fr", "es", "pt"]


def map_id_layer_to_level(ids_each_level) -> Dict[int, int]:
    dict_layers = {}
    lengthes = [len(id_one_level) for id_one_level in ids_each_level]
    tag_id = 0
    for i, length_tmp in enumerate(lengthes):
        for j in range(length_tmp):
            dict_layers[tag_id] = i
            tag_id += 1
    return dict_layers


def _clean_str_for_logging(text: str):
    return re.sub("[^0-9a-zA-Z]+", "_", copy(text))


def _clean_results_for_logging(
    results: Dict[str, Dict[str, float]], prefix: str = ""
) -> Dict[str, float]:
    """clean names and prepare them for logging"""

    final_mlflow_outputs = {}
    for tagname, tagresults in results.items():
        for metric, score in tagresults.items():
            mlflow_name = f"{metric}_{tagname}_{prefix}"
            mlflow_name = _clean_str_for_logging(mlflow_name)
            final_mlflow_outputs[mlflow_name] = score

    return final_mlflow_outputs


def _clean_thresholds_for_logging(thresholds: Dict[str, float]):
    return {
        f"threshold_{_clean_str_for_logging(tagname)}": tagthreshold
        for tagname, tagthreshold in thresholds.items()
    }


def get_tagname_to_id(target) -> Dict[str, int]:
    """
    Assign id to each tag
    """
    tag_set = set()
    for tags_i in target:
        tag_set.update(tags_i)
    tagname_to_tagid = {tag: i for i, tag in enumerate(list(sorted(tag_set)))}
    return tagname_to_tagid


def _custom_eval(x) -> List:
    if str(x) == "nan":
        return []
    if str(x) == "[None]":
        return []
    if type(x) == list:
        return x
    else:
        return literal_eval(x)


def clean_rows(row: List[str]) -> List[str]:
    """
    1) Apply litteral evaluation
    2) keep unique values
    """
    return list(set(_custom_eval(row)))


def _flatten(t: List[List]) -> List:
    """flatten list of lists"""
    return [item for sublist in t for item in sublist]


def _custom_stratified_ids_split(
    df: pd.DataFrame,
    ratios: Dict[str, float] = {"train": 0.9, "test": 0.1},
    tags_column_name: str = "target",
) -> Tuple[List[int], List[int]]:
    """
    custom function for stratified train test splitting
    1) take unique sub-tags (example: ['Health'])
    2) For each unique subtag:
        i) take all indexes that have that specific subtag
        ii) split them randomly to train and test sets
    """
    train_ids = []
    val_ids = []
    positive_df = df.copy()
    positive_df[tags_column_name] = positive_df[tags_column_name].apply(str)
    ids = positive_df.groupby(tags_column_name)["entry_id"].agg(list).values
    unique_ids = [list(set(list_)) for list_ in ids]

    for ids_entry in unique_ids:

        n_ids_entry = len(ids_entry)
        if n_ids_entry == 1:
            n_train_ids = 1
        elif n_ids_entry < 10:
            n_train_ids = n_ids_entry - 1
        else:
            n_train_ids = int(len(ids_entry) * ratios["train"]) + 1

        train_ids_entry = random.sample(ids_entry, n_train_ids)
        val_ids_entry = list(set(ids_entry) - set(train_ids_entry))

        train_ids.append(train_ids_entry)
        val_ids.append(val_ids_entry)

    return _flatten(train_ids), _flatten(val_ids)


def _preprocess_df(df: pd.DataFrame, min_entries_per_proj: int) -> pd.DataFrame:
    """
    main preprocessing function:
    1) get positive entries using the porportions of train test split
    2) add negative exapmles using the ratios defined in the training notebook

    NB: work with ids because the augmented sentences have the same entry_id as the original ones
    """
    # rename column to 'target' to be able to work on it generically
    all_data = df.copy()

    all_data = all_data[all_data.original_language.isin(languages)]

    all_data["target"] = all_data["target"].apply(_custom_eval)

    # drop duplicate entries
    all_data = all_data.groupby("en", as_index=False).agg(
        {
            "fr": lambda x: list(x)[0],
            "es": lambda x: list(x)[0],
            "pt": lambda x: list(x)[0],
            "target": lambda x: list(
                set.intersection(*map(set, list(x)))
            ),  # only elements in common between duplicates
            "original_language": lambda x: list(x)[0],
            "project_id": lambda x: list(x)[
                0
            ],  # doesn't matter which project is chosen
            "entry_id": lambda x: list(x)[0],  # doesn't matter which entry_id is chosen
        }
    )

    # delete entries with too many tags (sectors and subpillars): noise
    all_data = all_data[
        all_data["target"].apply(
            lambda x: len(
                [
                    item
                    for item in x
                    if any(
                        [
                            kw in item
                            for kw in ["first_level_tags->sectors", "subpillars"]
                        ]
                    )
                ]
            )
            <= 5
        )
    ]

    project_tags_df = (
        all_data.groupby("project_id", as_index=False)
        .agg(
            {
                "target": lambda x: list(set(_flatten(list(x)))),
                "entry_id": lambda x: len(list(x)),
            }
        )
        .rename(columns={"entry_id": "n_entries"})
    )

    project_used_for_training = project_tags_df[
        project_tags_df.n_entries > min_entries_per_proj
    ].project_id.tolist()
    projects_for_out_of_context_testing = list(
        set(project_tags_df.project_id.tolist()) - set(project_used_for_training)
    )

    out_of_context_test_data = all_data[
        (all_data.project_id.isin(projects_for_out_of_context_testing)) & (all_data.original_language.isin(languages))
    ]
    out_of_context_test_data['excerpt'] = out_of_context_test_data.apply(lambda x: x[x["original_language"]], axis=1)

    # keep only data with enough projects
    all_data = all_data[all_data.project_id.isin(project_used_for_training)]

    project_tags_dict = dict(zip(project_tags_df.project_id, project_tags_df.target))

    all_target = list(set(_flatten(project_tags_df["target"])))

    # Dict[str: tags, List[int]: projects list]
    projects_list_per_tag = {
        tag: [
            proj
            for proj, tags_per_proj in project_tags_dict.items()
            if tag in tags_per_proj
        ]
        for tag in all_target
    }

    return (all_data, projects_list_per_tag, out_of_context_test_data)


def create_train_val_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    main preprocessing function:
    1) get positive entries using the porportions of train test split
    2) add negative exapmles using the ratios defined in the training notebook

    NB: work with ids because the augmented sentences have the same entry_id as the original ones
    """
    # rename column to 'target' to be able to work on it generically
    dataset = df.dropna().copy()

    dataset["target"] = dataset.target.apply(lambda x: clean_rows(x))

    ratios = {
        "train": 0.85,
        "val": 0.15,
    }

    train_pos_entries, val_pos_entries = _custom_stratified_ids_split(dataset, ratios)

    df_train = dataset[dataset.entry_id.isin(train_pos_entries)]
    df_val = dataset[dataset.entry_id.isin(val_pos_entries)]

    return df_train, df_val


def stats_train_test(
    df_train: pd.DataFrame, df_val: pd.DataFrame, column_name: str
) -> float:
    """
    Sanity check of data (proportion negative examples)
    """

    def compute_ratio_negative_positive(df):
        nb_rows_negative = df[df.target.apply(lambda x: len(x) == 0)].shape[0]
        if len(df) > 0:
            return np.round(nb_rows_negative / df.shape[0], 2)
        else:
            return 0

    ratio_negative_positive = {
        f"ratio_negative_examples_train_{column_name}": compute_ratio_negative_positive(
            df_train
        ),
        f"ratio_negative_examples_val_{column_name}": compute_ratio_negative_positive(
            df_val
        ),
    }

    return ratio_negative_positive


def get_tags_proportions(
    tagname_to_tagid: Dict[str, int], targets_list: List[List[str]]
) -> torch.Tensor:
    """get alphas for BCE weighted loss"""
    counts = dict(Counter(_flatten(targets_list)))
    sorted_counts = [counts[k] for k, v in tagname_to_tagid.items()]
    return torch.tensor(
        compute_weights(number_data_classes=sorted_counts, n_tot=len(targets_list)),
        dtype=torch.float64,
    )


def compute_weights(number_data_classes: List[int], n_tot: int) -> List[float]:
    """
    weights computation for weighted loss function
    INPUTS:
    1) number_data_classes: list: number of samples for each class
    2) n_tot: total number of samples

    OUTPUT:
    list of weights used for training
    """
    return [number_data_class / n_tot for number_data_class in number_data_classes]


def get_flat_labels(column_of_columns, tag_to_id: Dict[str, int], nb_subtags: int):
    matrix = [
        [1 if tag_to_id[i] in column else 0 for i in range(nb_subtags)]
        for column in column_of_columns
    ]
    return np.array(_flatten(matrix))


def get_tag_id_to_layer_id(ids_each_level):
    tag_id = 0
    list_id = 0
    tag_to_list = {}
    for id_list in ids_each_level:
        for i in range(len(id_list)):
            tag_to_list.update({tag_id + i: list_id})
        tag_id += len(id_list)
        list_id += 1
    return tag_to_list


def get_first_level_ids(tagname_to_tagid: Dict[str, int]) -> List[List[List[int]]]:
    """having list of unique labels, create the labels ids in different lists"""
    all_names = list(tagname_to_tagid.keys())
    split_names = [name.split("->") for name in all_names]

    assert np.all([len(name_list) == 3 for name_list in split_names])
    final_ids = []

    tag_id = 0
    first_level_names = list(np.unique([name_list[0] for name_list in split_names]))
    for first_level_name in first_level_names:
        first_level_ids = []
        kept_names = [
            name_list[1:]
            for name_list in split_names
            if name_list[0] == first_level_name
        ]
        second_level_names = list(np.unique([name[0] for name in kept_names]))
        for second_level_name in second_level_names:
            second_level_ids = []
            third_level_names = [
                name_list[1]
                for name_list in kept_names
                if name_list[0] == second_level_name
            ]
            for _ in range(len(third_level_names)):
                second_level_ids.append(tag_id)
                tag_id += 1
            first_level_ids.append(second_level_ids)
        final_ids.append(first_level_ids)

    return final_ids


def get_relevant_labels(target_column, min_kept_ratio: float = 0.02) -> List[str]:
    n_items = len(target_column)
    targets = _flatten(
        target_column.apply(
            lambda x: [item for item in _custom_eval(x) if item != "NOT_MAPPED"]
        ).tolist()
    )
    relevant_labels = [
        label_name
        for label_name, label_counts in dict(Counter(targets)).items()
        if (label_counts / n_items) > min_kept_ratio
    ]
    return relevant_labels


def _get_n_tokens(
    text: List[str], tokenizer_name: str, batch_size_tokenizer: int = 128
) -> np.ndarray:
    """
    get number of tokens after tokeniziation for excerpts.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    masks = []
    for i in range(0, len(text), batch_size_tokenizer):

        one_batch = text[i : i + batch_size_tokenizer]

        masks.append(
            tokenizer(
                one_batch,
                None,
                truncation=False,
                add_special_tokens=True,
                padding="max_length",
                return_token_type_ids=False,
            )["attention_mask"]
        )

    lengths = np.array(
        [
            np.sum(np.array(one_mask) == 1)
            for masks_sublist in masks
            for one_mask in masks_sublist
        ]
    )

    return lengths


def hypertune_threshold(model, val_data, f_beta: float = 0.8):
    """
    having the probabilities, loop over a list of thresholds to see which one:
    1) yields the best results
    2) without being an aberrant value
    """

    logit_predictions, y_true = model.custom_predict(
        val_data,
        return_transformer_only=False,
        hypertuning_threshold=True,
    )

    optimal_thresholds_dict = {}
    optimal_scores = defaultdict(dict)

    for j in range(logit_predictions.shape[1]):
        preds_one_column = logit_predictions[:, j]
        min_proba = np.round(
            max(0.01, min(preds_one_column)), 3
        )  # so no value equal to 0
        max_proba = np.round(
            max(0.01, max(preds_one_column)), 3
        )  # so no value equal to 0

        thresholds_list = np.round(np.linspace(max_proba, min_proba, 21), 3)

        f_beta_scores = []
        precision_scores = []
        recall_scores = []
        for thresh_tmp in thresholds_list:
            score = get_metrics(
                np.array(preds_one_column > thresh_tmp).astype(int),
                np.array(y_true[:, j]),
                f_beta,
            )
            f_beta_scores.append(score["f_score"])
            precision_scores.append(score["precision"])
            recall_scores.append(score["recall"])

        max_threshold = 0.01
        best_f_beta_score = -1
        best_recall = -1
        best_precision = -1

        for i in range(1, len(f_beta_scores) - 1):

            f_beta_score_mean = np.mean(f_beta_scores[i - 1 : i + 2])
            precision_score_mean = np.mean(precision_scores[i - 1 : i + 2])
            recall_score_mean = np.mean(recall_scores[i - 1 : i + 2])

            if (
                f_beta_score_mean >= best_f_beta_score
                and abs(recall_score_mean - precision_score_mean) < 0.4
            ):

                best_f_beta_score = f_beta_score_mean
                best_recall = recall_score_mean
                best_precision = precision_score_mean

                max_threshold = thresholds_list[i]

        tag_name = list(model.tagname_to_tagid.keys())[j]

        optimal_scores[tag_name]["f_beta_score"] = best_f_beta_score
        optimal_scores[tag_name]["precision"] = best_precision
        optimal_scores[tag_name]["recall"] = best_recall

        optimal_thresholds_dict[tag_name] = max_threshold

    return optimal_thresholds_dict, optimal_scores


def _generate_results(
    predictions: List[List[str]],
    groundtruth: List[List[str]],
    tagname_to_tagid: Dict[str, int],
):

    n_entries = len(predictions)
    n_tags = len(tagname_to_tagid)
    binary_outputs_predictions = np.zeros((n_entries, n_tags))
    binary_outputs_groundtruth = np.zeros((n_entries, n_tags))

    for i in range(n_entries):
        preds_one_entry = set(predictions[i])
        groundtruth_one_entry = set(groundtruth[i])

        for tagname, tagid in tagname_to_tagid.items():
            if tagname in preds_one_entry:
                binary_outputs_predictions[i, tagid] = 1

            if tagname in groundtruth_one_entry:
                binary_outputs_groundtruth[i, tagid] = 1

    tot_scores = {}

    for tagname, tagid in tagname_to_tagid.items():
        predictions_one_tag = binary_outputs_predictions[:, tagid]
        groundtruths_one_tag = binary_outputs_groundtruth[:, tagid]

        scores_one_tag = get_metrics(predictions_one_tag, groundtruths_one_tag)

        tot_scores[tagname] = scores_one_tag

    return tot_scores


def get_metrics(preds: List[int], groundtruth: List[int], f_beta=1):
    """
    metrics for one tag
    """

    precision, recall, f_score, _ = precision_recall_fscore_support(
        groundtruth, preds, average="binary", beta=f_beta
    )

    confusion_results = confusion_matrix(groundtruth, preds, labels=[0, 1])
    n_test_set_excerpts = sum(sum(confusion_results))
    accuracy = (confusion_results[0, 0] + confusion_results[1, 1]) / n_test_set_excerpts
    sensitivity = confusion_results[0, 0] / (
        confusion_results[0, 0] + confusion_results[0, 1]
    )
    specificity = confusion_results[1, 1] / (
        confusion_results[1, 0] + confusion_results[1, 1]
    )

    return {
        "precision": np.round(precision, 3),
        "recall": np.round(recall, 3),
        "f_score": np.round(f_score, 3),
        "accuracy": np.round(accuracy, 3),
        "sensitivity": np.round(sensitivity, 3),
        "specificity": np.round(specificity, 3),
    }


def _create_df_with_translations(df: pd.DataFrame):
    """
    create final df, with translations
    """

    full_data = df.copy()
    augmented_data = pd.DataFrame()

    for one_lang in languages:
        one_lang_df = (
            full_data[["entry_id", one_lang]]
            .rename(columns={one_lang: "excerpt"})
            .dropna()
        )
        augmented_data = pd.concat([augmented_data, one_lang_df])

    augmented_data = pd.merge(
        left=augmented_data,
        right=full_data[["entry_id", "project_id", "target"]].copy(),
        on="entry_id",
    )

    return augmented_data


def _create_df_with_chosen_translations(data: pd.DataFrame):
    """
    not including all ranslations, but a way to balance data:
        too represented tags: deleting excerpts (undersampling)
        midrepresented tags: no translation
        underrepresented tags: full translation
    """
    df = data.copy()

    targets_list = df["target"].tolist()
    tagname_to_tagid = get_tagname_to_id(targets_list)
    proportions = get_tags_proportions(tagname_to_tagid, targets_list)
    tags_proportions = {
        tagname: proportions[tagid].item()
        for tagname, tagid in tagname_to_tagid.items()
    }

    min_prop = np.quantile(proportions, 0.75)

    underrepresented_tags = [
        tagname for tagname, prop in tags_proportions.items() if prop < min_prop
    ]

    mask_df_contains_underrepresented_tag = df.target.apply(
        lambda x: any(tag in underrepresented_tags for tag in x)
    )

    df_with_low_entries = df[mask_df_contains_underrepresented_tag]
    augmented_df_with_low_entries = _create_df_with_translations(df_with_low_entries)

    df_without_low_entries = df[~mask_df_contains_underrepresented_tag]
    df_without_low_entries["excerpt"] = df_without_low_entries.apply(
        lambda x: _get_excerpt_without_augmentation(x),
        axis=1,
    )

    return pd.concat([augmented_df_with_low_entries, df_without_low_entries])[
        ["entry_id", "project_id", "target", "excerpt"]
    ]


def _get_excerpt_without_augmentation(row: pd.Series):
    return (
        row[row["original_language"]]
        if row["original_language"] in languages
        else row["en"]
    )


def _undersample_df(
    df: pd.DataFrame,
    tags_proportions: Dict[str, float],
):
    max_prop = 10 * np.median(list(tags_proportions.values()))
    overrepresented_tags = [
        tagname for tagname, prop in tags_proportions.items() if prop > max_prop
    ]

    mask_df_contains_overrepresented_tag = df.target.apply(
        lambda x: all(tag in overrepresented_tags for tag in x)
    )

    undersampled_df = df[~mask_df_contains_overrepresented_tag]

    return undersampled_df


def _get_labled_unlabled_data(
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    projects_list_one_same_tags_set: List[int],
    tags_with_same_projects: List[str],
):
    mask_labeled_projects = train_val_df.project_id.isin(
        projects_list_one_same_tags_set
    )

    # train val data: labeled
    train_val_data_labeled = train_val_df[mask_labeled_projects].copy()
    train_val_data_labeled["target"] = train_val_data_labeled["target"].apply(
        lambda x: [tag for tag in x if tag in tags_with_same_projects]
    )
    # take care of translation for train val labeled
    train_val_data_labeled = _create_df_with_chosen_translations(train_val_data_labeled)

    # train val data: to be relabeled
    train_val_data_one_tag_non_labeled = train_val_df[~mask_labeled_projects].copy()

    if len(train_val_data_one_tag_non_labeled) > 0:
        train_val_data_one_tag_non_labeled[
            "excerpt"
        ] = train_val_data_one_tag_non_labeled.apply(
            lambda x: x[x["original_language"]], axis=1
        )

    # test set for results generation
    test_data_one_tag_labeled = test_df[
        test_df.project_id.isin(projects_list_one_same_tags_set)
    ]
    test_data_one_tag_labeled["target"] = test_data_one_tag_labeled["target"].apply(
        lambda x: [tag for tag in x if tag in tags_with_same_projects]
    )

    return (
        train_val_data_labeled,
        train_val_data_one_tag_non_labeled,
        test_data_one_tag_labeled,
    )


def _create_stratified_train_test_df(all_data: pd.DataFrame, sample_data: bool = False):
    all_data = all_data[all_data.original_language.isin(["en", "fr", "pt", "es"])]
    train_val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    all_projects = all_data.project_id.unique().tolist()
    for one_project_id in all_projects:
        data_one_project = all_data[all_data.project_id == one_project_id].copy()
        (
            train_val_ids_one_project,
            test_ids_one_project,
        ) = _custom_stratified_ids_split(data_one_project, tags_column_name="target")
        train_val_df = pd.concat(
            [
                train_val_df,
                data_one_project[
                    data_one_project.entry_id.isin(train_val_ids_one_project)
                ],
            ]
        )
        test_df = pd.concat(
            [
                test_df,
                data_one_project[data_one_project.entry_id.isin(test_ids_one_project)],
            ]
        )

    if len(test_df) > 0:
        test_df["excerpt"] = test_df.apply(lambda x: x[x["original_language"]], axis=1)

    if sample_data:
        train_val_df = train_val_df.sample(n=200)
        test_df = test_df.sample(n=10)

    return train_val_df, test_df


def _get_parent_tags(all_tags: List[str]):

    # from mapping sheet
    parent_tags = defaultdict(list)
    parent_tags.update(
        {
            "secondary_tags->Age->12-17 years old": [
                "secondary_tags->Age-><18 years old",
                "secondary_tags->Age->5-17 years old",
            ],
            "secondary_tags->Age->5-11 years old": [
                "secondary_tags->Age-><18 years old",
                "secondary_tags->Age->5-17 years old",
            ],
            "secondary_tags->Age->18-24 years old": [
                "secondary_tags->Age->18-59 years old"
            ],
            "secondary_tags->Age->25-59 years old": [
                "secondary_tags->Age->18-59 years old"
            ],
            "secondary_tags->Age->5-17 years old": [
                "secondary_tags->Age-><18 years old"
            ],
            "secondary_tags->Age-><5 years old": ["secondary_tags->Age-><18 years old"],
            "secondary_tags->Displaced->In transit": [
                "secondary_tags->Displaced->Others of concern"
            ],
            "secondary_tags->Displaced->Irregular": [
                "secondary_tags->Displaced->Migrants"
            ],
            "secondary_tags->Displaced->Regular": [
                "secondary_tags->Displaced->Migrants"
            ],
            "secondary_tags->Displaced->Pendular": [
                "secondary_tags->Displaced->Migrants"
            ],
        }
    )

    for tag in all_tags:
        if "secondary_tags->Displaced" in tag:
            parent_tags[tag].append("first_level_tags->Affected->Displaced")
        elif "secondary_tags->Non displaced" in tag:
            parent_tags[tag].append("first_level_tags->Affected->Non displaced")
        elif any(
            [kw in tag for kw in ["subpillars_2d", "subpillars_1d", "subsectors"]]
        ):
            parent_kw = "->".join(tag.replace("sub", "").split("->")[:-1])
            parent_kw = f"first_level_tags->{parent_kw}"
            parent_tags[tag].append(parent_kw)

    return parent_tags


def _tag_is_valid(tag: str, tags_results: str, parent_tags: Dict[str, List[str]]):
    """
    tag: each prediction
    tags_results: all predictions
    parent tags: all parents of each tag need to be there to validate tag
    """
    if tag not in parent_tags:
        return True
    else:
        tag_parents = parent_tags[tag]
        if all([one_parent_tag in tags_results for one_parent_tag in tag_parents]):
            return True
        else:
            return False


def _get_final_demographic_groups(demographic_group_tags: List[str]):

    if len(demographic_group_tags) == 0:
        return []

    potprocessed_demographic_groups = []
    gender_tag = [tag for tag in demographic_group_tags if "Gender" in tag]
    assert len(gender_tag) <= 1, "problem in tags postprocessing"

    if len(gender_tag) == 0:
        final_gender_tag = "secondary_tags->Gender->All"
    else:
        final_gender_tag = gender_tag[0]

    age_tags = [tag for tag in demographic_group_tags if "Gender" in tag]
    if len(age_tags) == 0:
        potprocessed_demographic_groups.append(
            f"secondary_tags->Demographic Groups->{final_gender_tag.capitalize()}"
        )
    else:
        potprocessed_demographic_groups.extend(
            [
                f"secondary_tags->Demographic Groups->{final_gender_tag.capitalize()} - {one_age.capitalize()}"
                for one_age in age_tags
            ]
        )

    return potprocessed_demographic_groups


def _postprocess_predictions_one_excerpt(
    tag_results: Dict[str, float],
    all_postprocessed_labels: List[str],
    single_label_tags: List[List[str]],
    parent_tags: Dict[str, List[str]],
):
    """
    tag_results: Dict[str: tag name, float: ratio of predictions]
    single_label_tags: groups of labels to be treated
    all_postprocessed_labels: flatened single_label_tags
    """
    tmp_outputs = {
        tag: proportion
        for tag, proportion in tag_results.items()
        if tag not in all_postprocessed_labels
    }
    for one_tags_list in single_label_tags:
        output_one_list = {
            tag: proportion
            for tag, proportion in tag_results.items()
            if tag in one_tags_list
        }
        if len(output_one_list) > 0:
            output_one_list = max(output_one_list, key=output_one_list.get)
            tmp_outputs.update({output_one_list: tag_results[output_one_list]})

    tmp_outputs = [
        tag
        for tag, prediction_ratio in tmp_outputs.items()
        if prediction_ratio >= 1 or "severity" in tag
    ]
    tmp_outputs = [
        tag for tag in tmp_outputs if _tag_is_valid(tag, tmp_outputs, parent_tags)
    ]
    final_outputs = [
        tag for tag in tmp_outputs if all([kw not in tag for kw in ["Age", "Gender"]])
    ]
    demographic_group_outputs = [
        tag for tag in tmp_outputs if all([kw in tag for kw in ["Age", "Gender"]])
    ]
    final_outputs.extend(_get_final_demographic_groups(demographic_group_outputs))

    return final_outputs


def _get_sectors_non_sectors_grouped_tags(grouped_tags: List[List[str]]):
    non_sector_groups, sector_groups = [], []
    sectors_kw = "first_level_tags->sectors"
    not_relabled_kwords = ["cross", "covid-19", "severity"]

    for one_group_tags in grouped_tags:
        one_group_non_sectors, one_group_sectors = [], []
        for one_tag in one_group_tags:
            if all([one_kw not in one_tag.lower() for one_kw in not_relabled_kwords]):
                if sectors_kw in one_tag:
                    one_group_sectors.append(one_tag)
                else:
                    one_group_non_sectors.append(one_tag)

        if len(one_group_non_sectors) > 0:
            non_sector_groups.append(one_group_non_sectors)
        if len(one_group_sectors) > 0:
            sector_groups.append(one_group_sectors)

    return sector_groups, non_sector_groups


def _update_final_labels_dict(
    train_val_data_labeled: pd.DataFrame,
    final_predictions_unlabled_train_val: List[List[str]],
    train_val_data_non_labeled: pd.DataFrame,
    train_val_final_labels: Dict[str, List[str]],
):

    # update with groundtruths labels
    for i, row in train_val_data_labeled.iterrows():
        train_val_final_labels[row["entry_id"]].extend(row["target"])

    # update with newly labeled data
    for i in range(len(final_predictions_unlabled_train_val)):
        train_val_final_labels[train_val_data_non_labeled.iloc[i]["entry_id"]].extend(
            final_predictions_unlabled_train_val[i]
        )

    return train_val_final_labels


def _proportion_false_negatives(
    train_val_data_labeled: pd.DataFrame,
    train_val_data_non_labeled: pd.DataFrame,
):
    """
    Estimate proportion of false positives in dataset.
    """
    if len(train_val_data_non_labeled) > 0:
        n_ids_train_val_labeled = train_val_data_labeled.entry_id.nunique()
        n_pos_examples_train_val_labeled = train_val_data_labeled.target.apply(
            lambda x: len(x) > 0
        ).sum()
        prop_pos_examples_train_val_labeled = (
            n_pos_examples_train_val_labeled / n_ids_train_val_labeled
        )

        n_ids_train_val_non_labeled = train_val_data_non_labeled.entry_id.nunique()
        estimated_false_negatives = (
            prop_pos_examples_train_val_labeled * n_ids_train_val_non_labeled
        )

        tot_train_val = n_ids_train_val_labeled + n_ids_train_val_non_labeled

        final_prop = estimated_false_negatives / tot_train_val
    else:
        final_prop = -1

    return final_prop


def _get_new_sectors_tags(
    row: pd.Series,
    cross_targets: Dict[int, List[str]],
    non_trained_data: Dict[
        int, Dict[int, List[str]]
    ],  # {project_id: {entry_id: targets: List[str]}}
    projects_list_per_tag: Dict[str, List[int]],
    sector_tags: List[str],
):

    row_id = row.entry_id
    row_project = row.project_id
    row_target_non_sectors = [
        item for item in row["target"] if "first_level_tags->sectors->" not in item
    ]
    original_sector_labels = [
        item
        for item in row["target"]
        if "first_level_tags->sectors->" in item and "Cross" not in item
    ]
    if row_id in cross_targets:
        sector_labels = cross_targets[row_id]
    elif row_project in non_trained_data:

        relabled_sector_labels = non_trained_data[row_project][row_id]
        sector_labels = []
        for one_sector in sector_tags:
            if (
                row_project in projects_list_per_tag[one_sector]
                and one_sector in original_sector_labels
            ):
                sector_labels.append(one_sector)
            elif one_sector in relabled_sector_labels:
                sector_labels.append(one_sector)

    else:
        sector_labels = original_sector_labels

    return list(set(sector_labels + row_target_non_sectors))


def _get_new_subsectors_tags(
    row: pd.Series,
    sector_name: str,
    predictions: Dict[int, List[str]],
):
    """
    After generating predictions, integrate them in final labels
    Only include predictions of subsectors in sectors that have been already tagged
    Example:
        prediction is 'subsectors->Wash->Vector Control'
        We only include it if there is the tag 'first_level_tags->sectors->Wash'
    """
    # predictions: dict entry_id to predicted target
    row_target = row["target"]
    row_id = row["entry_id"]
    final_predictions = copy(row_target)
    if row_id in predictions and sector_name in row_target:
        entry_predictions = predictions[row_id]
        final_predictions.extend(entry_predictions)

    return list(set(final_predictions))


def _generate_test_set_results(
    transformer_model,
    test_data: pd.DataFrame,
    projects_list_per_tag: Dict[str, List[int]],
):
    """
    Generate test set results
    1- Generate predictions and results on labeled test set
    2- Get results on test set of project that contain the tag (to avoid having false negatives)
    For sector tags: No entries where there is 'Cross'.
    """
    # Generate predictions and results on labeled test set
    final_results = {}
    test_df = test_data.copy()
    test_df["predictions"] = transformer_model.generate_test_predictions(
        test_df.excerpt.tolist(), apply_postprocessing=False
    )

    for name, projects in projects_list_per_tag.items():
        test_df_one_tag = test_df[test_df.project_id.isin(projects)].copy()
        if "first_level_tags->sector" not in name:
            test_df_one_tag = test_df_one_tag[
                test_df_one_tag.target.apply(
                    lambda x: "first_level_tags->sectors->Cross" not in x
                )
            ]
        if len(test_df_one_tag) > 0:
            results_one_tag = _generate_results(
                test_df_one_tag["predictions"].tolist(),
                test_df_one_tag.target.tolist(),
                transformer_model.tagname_to_tagid,
            )
            results_one_tag = {
                tagname: tagresults
                for tagname, tagresults in results_one_tag.items()
                if name == tagname
            }
            final_results.update(results_one_tag)

    return final_results


def _get_results_df_from_dict(
    final_results: Dict[str, Dict[str, float]], proportions: Dict[str, float]
):
    """
    input: Dict: {tagname: {metric: score}}
    output: results as a dataframe and mean outputs of each tag
    """
    results_as_df = pd.DataFrame.from_dict(final_results, orient="index")
    metrics_list = list(results_as_df.columns)
    results_as_df["tag"] = results_as_df.index
    results_as_df.sort_values(by=["tag"], inplace=True, ascending=True)
    results_as_df["positive_examples_proportion"] = [
        proportions[one_tag].item() for one_tag in results_as_df["tag"]
    ]

    # get mean results
    mean_results_df = results_as_df.copy()
    mean_results_df["tag"] = mean_results_df["tag"].apply(
        lambda x: "mean->" + "->".join(x.split("->")[:-1])
    )
    mean_results_df = mean_results_df.groupby("tag", as_index=False).agg(
        {metric: lambda x: np.mean(list(x)) for metric in metrics_list}
    )
    mean_results_df["positive_examples_proportion"] = "-"
    results_as_df = pd.concat([results_as_df, mean_results_df])

    ordered_columns = ["tag"] + metrics_list + ["positive_examples_proportion"]
    results_as_df = results_as_df[ordered_columns].round(
        {col: 3 for col in ordered_columns if col != "tag"}
    )

    return results_as_df


######################## VIZUALIZATION ##############################


def _get_bar_colour(q: float):
    """
    different colour code depending on score.
    """
    if q < 0.3:
        palette_colour_code = "#90e0ef"
    elif q < 0.6:
        palette_colour_code = "#0077b6"
    else:
        palette_colour_code = "#03045e"
    return palette_colour_code


def _get_handles():
    """
    handles, used for tag results colors visualization
    """
    usually_reliable_tags = mpatches.Patch(
        color="#03045e", label="strong results (f1 score > 0.6)"
    )
    fairly_reliable_tags = mpatches.Patch(
        color="#0077b6", label="challenging results (0.3 < f1 score < 0.6"
    )
    not_usually_reliable_tags = mpatches.Patch(
        color="#90e0ef", label="room for improvement results (f1 score < 0.3)"
    )

    # unreliable_tags = mpatches.Patch(color='#caf0f8', label='unreliable tags')
    handles = [
        usually_reliable_tags,
        fairly_reliable_tags,
        not_usually_reliable_tags,
    ]

    return handles
