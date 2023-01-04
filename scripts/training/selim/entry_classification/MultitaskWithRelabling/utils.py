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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

# GENERAL UTIL FUNCTIONS


def map_id_layer_to_level(ids_each_level) -> Dict[int, int]:
    dict_layers = {}
    lengthes = [len(id_one_level) for id_one_level in ids_each_level]
    tag_id = 0
    for i, length_tmp in enumerate(lengthes):
        for j in range(length_tmp):
            dict_layers[tag_id] = i
            tag_id += 1
    return dict_layers


def beta_score(precision: float, recall: float, f_beta: Union[int, float]) -> float:
    """get beta score from precision and recall"""
    return (1 + f_beta**2) * precision * recall / ((f_beta**2) * precision + recall)


def clean_str_for_logging(text: str):
    return re.sub("[^0-9a-zA-Z]+", "_", copy(text))


def clean_results_for_logging(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """clean names and prepare them for logging"""

    final_mlflow_outputs = {}
    for tagname, tagresults in results.items():
        for metric, score in tagresults.items():
            mlflow_name = f"{metric}_{tagname}"
            mlflow_name = clean_str_for_logging(mlflow_name)
            final_mlflow_outputs[mlflow_name] = score

    return final_mlflow_outputs


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


def preprocess_df(df: pd.DataFrame, min_entries_per_proj: int) -> pd.DataFrame:
    """
    main preprocessing function:
    1) get positive entries using the porportions of train test split
    2) add negative exapmles using the ratios defined in the training notebook

    NB: work with ids because the augmented sentences have the same entry_id as the original ones
    """
    # rename column to 'target' to be able to work on it generically
    all_data = df.copy()

    # only unprotected leads for training
    all_data = all_data[all_data.confidentiality == "unprotected"]

    all_data["nlp_tags"] = all_data["nlp_tags"].apply(_custom_eval)

    # drop duplicate entries
    all_data = all_data.groupby("en", as_index=False).agg(
        {
            "fr": lambda x: list(x)[0],
            "es": lambda x: list(x)[0],
            "pt": lambda x: list(x)[0],
            "nlp_tags": lambda x: list(
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
        all_data["nlp_tags"].apply(
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
                "nlp_tags": lambda x: list(set(_flatten(list(x)))),
                "entry_id": lambda x: len(list(x)),
            }
        )
        .rename(columns={"entry_id": "n_entries"})
    )

    project_tags_df = project_tags_df[project_tags_df.n_entries > min_entries_per_proj]

    # keep only data with enough projects
    all_data = all_data[all_data.project_id.isin(project_tags_df.project_id.tolist())]

    project_tags_dict = dict(zip(project_tags_df.project_id, project_tags_df.nlp_tags))

    all_nlp_tags = list(set(_flatten(project_tags_df["nlp_tags"])))

    # Dict[str: tags, List[int]: projects list]
    projects_list_per_tag = {
        tag: [
            proj
            for proj, tags_per_proj in project_tags_dict.items()
            if tag in tags_per_proj
        ]
        for tag in all_nlp_tags
    }

    grouped_tags = {k: str(v) for k, v in projects_list_per_tag.items()}
    tmp_dict = defaultdict(list)
    for k, v in grouped_tags.items():
        tmp_dict[v].append(k)

    # List[List[str]: each sublist has exactly the same project_ids]
    grouped_tags = list(tmp_dict.values())

    all_data.rename(columns={"nlp_tags": "target"}, inplace=True)

    return all_data, projects_list_per_tag, grouped_tags


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
    tagname_to_tagid: Dict[str, int], targets_list: List[str]
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


def get_n_tokens(
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

    logit_predictions, y_true, _ = model.custom_predict(
        val_data,
        return_transformer_only=False,
        hypertuning_threshold=True,
    )

    optimal_thresholds_dict = {}
    optimal_f_beta_scores = {}
    optimal_precision_scores = {}
    optimal_recall_scores = {}

    for j in range(logit_predictions.shape[1]):
        preds_one_column = logit_predictions[:, j]
        min_proba = np.round(
            max(0.01, min(preds_one_column)), 2
        )  # so no value equal to 0
        max_proba = np.round(
            max(0.01, max(preds_one_column)), 2
        )  # so no value equal to 0

        thresholds_list = np.round(np.linspace(max_proba, min_proba, 21), 2)

        f_beta_scores = []
        precision_scores = []
        recall_scoress = []
        for thresh_tmp in thresholds_list:
            score = get_metrics(
                np.array(preds_one_column > thresh_tmp).astype(int),
                np.array(y_true[:, j]),
                f_beta,
            )
            f_beta_scores.append(score["f_beta_score"])
            precision_scores.append(score["precision"])
            recall_scoress.append(score["recall"])

        max_threshold = 0.01
        best_f_beta_score = -1
        best_recall = -1
        best_precision = -1

        for i in range(2, len(f_beta_scores) - 2):

            f_beta_score_mean = np.mean(f_beta_scores[i - 2 : i + 2])
            precision_score_mean = np.mean(precision_scores[i - 2 : i + 2])
            recall_score_mean = np.mean(recall_scoress[i - 2 : i + 2])

            if f_beta_score_mean >= best_f_beta_score:

                best_f_beta_score = f_beta_score_mean
                best_recall = recall_score_mean
                best_precision = precision_score_mean

                max_threshold = thresholds_list[i]

        tag_name = list(model.tagname_to_tagid.keys())[j]

        optimal_f_beta_scores[tag_name] = best_f_beta_score
        optimal_precision_scores[tag_name] = best_precision
        optimal_recall_scores[tag_name] = best_recall

        optimal_thresholds_dict[tag_name] = max_threshold

    optimal_scores = {
        "precision": optimal_precision_scores,
        "recall": optimal_recall_scores,
        "f_beta_scores": optimal_f_beta_scores,
    }

    return optimal_thresholds_dict, optimal_scores


def generate_results(model, entries: List[str], groundtruth: List[List[str]]):
    predictions = model.generate_test_predictions(entries)
    mlb = MultiLabelBinarizer(list(model.tagname_to_tagid.keys()))

    binary_outputs_predictions = mlb.fit_transform(predictions)
    binary_outputs_groundtruth = mlb.fit_transform(groundtruth)

    tot_scores = {}

    for tagname, tagid in model.tagname_to_tagid.items():
        predictions_one_tag = binary_outputs_predictions[:, tagid]
        groundtruths_one_tag = binary_outputs_groundtruth[:, tagid]

        scores_one_tag = get_metrics(predictions_one_tag, groundtruths_one_tag)

        tot_scores[tagname] = scores_one_tag

    return tot_scores


def get_metrics(preds, groundtruth, f_beta=1):

    precision, recall, f_beta_score, _ = precision_recall_fscore_support(
        groundtruth, preds, average="binary", beta=f_beta
    )

    return {
        "precision": np.round(precision, 3),
        "recall": np.round(recall, 3),
        "f_beta_score": np.round(f_beta_score, 3),
    }


def _create_df_with_translations(df: pd.DataFrame):

    full_data = df.copy()
    augmented_data = pd.DataFrame()
    languages = ["en", "fr", "es", "pt"]
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
    train_val_data_labeled = _create_df_with_translations(train_val_data_labeled.copy())

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

    return tag_results


def _get_sectors_non_sectors_grouped_tags(grouped_tags: List[List[str]]):
    non_sector_groups, sector_groups = [], []
    sectors_kw = "first_level_tags->sectors"
    cross_tag_name = "first_level_tags->sectors->Cross"

    for one_group_tags in grouped_tags:
        one_group_non_sectors, one_group_sectors = [], []
        for one_tag in one_group_tags:
            if one_tag != cross_tag_name:
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
