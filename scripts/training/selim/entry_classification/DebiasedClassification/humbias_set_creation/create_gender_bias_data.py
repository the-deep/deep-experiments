from string import punctuation
from typing import List, Dict
from copy import copy
import pandas as pd
from humbias_set_creation.utils import _extract_kwords, _replace_kw
from humbias_set_creation.gender_mappings.english_mappings import (
    en_biases_all,
    en_to_be_removed_kwords,
)
from humbias_set_creation.gender_mappings.french_mappings import (
    fr_biases_all,
    fr_to_be_removed_kwords,
)

biases_all = en_biases_all
to_be_removed_kwords = en_to_be_removed_kwords


def _flatten(t: List[List]) -> List:
    """flatten list of lists"""
    return [item for sublist in t for item in sublist]


def _generate_augmented_excerpt(excerpt: str, mapping: Dict[str, str]):
    new_excerpt = []
    excerpt_words = copy(excerpt).split(" ")
    for one_word in excerpt_words:
        clean_word = copy(one_word).lower()
        if "'s" in clean_word:
            one_word = one_word.replace("'s", "")
            clean_word = clean_word.replace("'s", "")
            added_str_in_end = "'s"
        else:
            added_str_in_end = ""

        clean_word = clean_word.translate(str.maketrans("", "", punctuation))
        if clean_word in mapping.keys():
            changed_word = mapping[clean_word]
            changed_word = _replace_kw(
                one_word, clean_word=clean_word, new_kw=changed_word
            )
        else:
            changed_word = one_word

        new_excerpt.append(f"{changed_word}{added_str_in_end}")

    return " ".join(new_excerpt)


def _augment_one_df(df, mapping):
    new_df = df.copy()
    # update kwords
    new_df["gender_keywords"] = new_df["gender_keywords"].apply(
        lambda x: [mapping[item] for item in x]
    )
    # update excerpt
    new_df["excerpt"] = new_df["excerpt"].apply(
        lambda x: _generate_augmented_excerpt(x, mapping)
    )
    return new_df


def _augment_gender_df(df: pd.DataFrame) -> pd.DataFrame:
    gender_df = df.copy()
    final_df = pd.DataFrame()
    for kw, all_related_to_kw in biases_all.items():
        one_df = gender_df[
            gender_df.gender_kword_type.apply(lambda x: x[0] == kw)
        ].copy()
        for new_kw, one_mapping in all_related_to_kw["mappings"].items():
            augmented_one_df = _augment_one_df(one_df, one_mapping)
            augmented_one_df["gender_kword_type"] = [
                [new_kw] for _ in range(len(augmented_one_df))
            ]
            augmented_one_df["excerpt_type"] = "augmented"
            final_df = pd.concat([final_df, augmented_one_df])

    return final_df


def _create_gender_df(df: pd.DataFrame) -> pd.DataFrame:
    gender_df = df.copy()

    gender_df["gender_context_falsing_kw"] = df.excerpt.apply(
        lambda x: [item for item in to_be_removed_kwords if item in x.lower()]
    )

    for kw in biases_all.keys():
        gender_df[f"{kw}_kwords"] = df.excerpt.apply(
            lambda x: _extract_kwords(x, biases_all[kw]["keywords"])
        )

    gender_df["gender_keywords"] = gender_df.apply(
        lambda x: _flatten([x[f"{kw}_kwords"] for kw in biases_all.keys()]),
        axis=1,
    )
    gender_df["gender_kword_type"] = gender_df.apply(
        lambda x: list(
            set([kw for kw in biases_all.keys() if len(x[f"{kw}_kwords"]) > 0])
        ),
        axis=1,
    )
    gender_df.drop(columns=[f"{kw}_kwords" for kw in biases_all.keys()], inplace=True)

    return gender_df


def create_gender_augmented_dataset(df):
    gender_df = _create_gender_df(df)

    # create augmented df
    augmented_df = _augment_gender_df(gender_df)
    augmented_df = pd.concat([gender_df, augmented_df])

    return augmented_df


def _get_mask_one_gender(df: pd.DataFrame) -> pd.DataFrame:
    mask_gender = (
        (df.gender_context_falsing_kw.apply(lambda x: len(x) == 0))
        & (df.gender_kword_type.apply(lambda x: len(x) == 1))
        & (df.original_language == "en")
    )

    return mask_gender


def _keep_one_gender_df(df: pd.DataFrame) -> pd.DataFrame:
    mask_gender = _get_mask_one_gender(df)
    final_df = df.copy()[mask_gender][
        [
            "entry_id",
            "excerpt",
            "gender_keywords",
            "gender_kword_type",
            "excerpt_type",
            "target",
            "original_language",
        ]
    ]

    return final_df
