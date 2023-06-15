from IPython.display import display
from typing import List, Dict
import pandas as pd

from humbias_set_creation.create_countries_bias_data import (
    create_country_augmented_dataset,
    _create_country_df,
    _keep_one_country_df,
    _get_mask_one_country,
    _augment_country_df,
)

from humbias_set_creation.create_gender_bias_data import (
    create_gender_augmented_dataset,
    _create_gender_df,
    _keep_one_gender_df,
    _get_mask_one_gender,
    _augment_gender_df,
)
from humbias_set_creation.utils import _create_scraped_excerpt

from humbias_set_creation.utils import (
    _clean_biases_dataset,
    _custom_eval,
)

BIASES_TYPES = ["gender", "country"]


def _get_genders_countries(df: pd.DataFrame) -> pd.DataFrame:
    classification_df = df.copy()  # .drop_duplicates()
    """
    1. clean excerpt
    2. extract gender keywords
    3. extract country keywords
    """

    classification_df["excerpt_type"] = "original"
    classification_df = _create_gender_df(classification_df)
    classification_df = _create_country_df(classification_df)

    return classification_df


def _initialize_test_counterfactual(
    test_df: pd.DataFrame, max_len: int, tokenizer
) -> Dict[str, pd.DataFrame]:
    gender_test_df = _keep_one_gender_df(test_df)
    country_test_df = _keep_one_country_df(test_df)

    # counterfactual
    gender_counterfactual = _clean_biases_dataset(
        create_gender_augmented_dataset(gender_test_df),
        max_len,
        tokenizer,
        task_name="gender",
    ).rename(columns={"gender_kword_type": "kword_type", "gender_keywords": "keywords"})
    # print("gender counterfactual test")
    # display(gender_counterfactual.head())
    country_counterfactual = _clean_biases_dataset(
        create_country_augmented_dataset(country_test_df),
        max_len,
        tokenizer,
        task_name="country",
    ).rename(
        columns={"country_kword_type": "kword_type", "country_keywords": "keywords"}
    )
    # print("country counterfactual test")
    # display(country_counterfactual.head())
    test_set_datasets = {
        "gender": gender_counterfactual,
        "country": country_counterfactual,
    }

    return test_set_datasets


def _initialize_counterfactual_dataset(
    df: pd.DataFrame,
):
    final_df = df.copy()
    for one_attribute in BIASES_TYPES:
        if one_attribute == "gender":
            mask_building_function = _get_mask_one_gender
            augmentation_function = _augment_gender_df

        elif one_attribute == "country":
            mask_building_function = _get_mask_one_country
            augmentation_function = _augment_country_df

        else:
            raise RuntimeError(f"attribute '{one_attribute}' not in {BIASES_TYPES}.")

        mask = mask_building_function(final_df)
        to_be_augmented_df = final_df[mask]

        augmented_df = augmentation_function(to_be_augmented_df)
        final_df = pd.concat([final_df, augmented_df])

    return final_df
