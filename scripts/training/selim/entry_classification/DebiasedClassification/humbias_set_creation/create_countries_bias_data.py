from humbias_set_creation.utils import _extract_kwords, _replace_kw
from copy import copy
import pandas as pd
from string import punctuation
from typing import List

############################### COUNTRIES AND NATIONALITIES CREATION UTILS #######################

# manually extracted syrian locations that may false predictions
syrian_location_specific_keywords = [
    "raqqa",
    "damascus",
    "tal-abiad",
    "ras al-ain",
    "tabqa",
    "deir-ez-zor",
    "al-hasakeh",
    "quneitra",
    "tartous",
    "hama",
    "as-sweida",
    "homs",
    "lattakia",
    "aleppo",
    "kufra",
]

# manually extracted venezuelan locations that may false predictions
venezuela_specific_locations = [
    "aruba",
    "curaçao",
    "medellín",
    "medellin",
    "boliv",
    "tucupita",
    "maduro",
    "guajira",
    "canciones",
    "cauca",
    "catatumbo",
    "villa del rosario",
]

aghanistan_specific_kwords = ["khost", "paktika", "waziri", "dawar", "saidgi", "masood"]


# most occurent nationalities (including syrian and venezuelan and canadian), extracted manually from our dataset
nationalities_kwords = [
    "bangladeshi",
    "brazilian",
    "congolese",
    "ecuadorian",
    "guatemalan",
    "honduran",
    "lebanese",
    "myanmarian",
    "nicaraguan",
    "nigerian",
    "peruvian",
    "sudanese",
    "uruguayan",
    "french",
    "italian",
    "swiss",
    "canadian",
    "ukrainian",
    "turkish",
    "chilian",
    "colombian",
    "libyan",
    "burundian",
    "swedish",
    "japanese",
    "caribbean",
    "american",
    "colombian",
    "trinidadian",
    "bolivari",
    "canadian",
    "syrian",
    "venezuelan",
]

# most occurent countries (not including syria and venezuela), extracted manuallly from our dataset
non_extracted_countries_kwords = [
    "argentina",
    "bangladesh",
    "brazil",
    "congo",
    "ecuador",
    "guatemala",
    "honduras",
    "lebanon",
    "mali",
    "myanmar",
    "nicaragua",
    "niger",
    "peru",
    "south sudan",
    "sudan",
    "uruguay",
    "france",
    "switzerland",
    "chile",
    "colombia",
    "libya",
    "burundi",
    "sweden",
    "russia",
    "us",
    "united states",
    "colombia",
    "bolivia",
    "chile",
    "tobago",
    "dominican republic",
    "guyana",
    "afghanistan",
    "canada",
    "greece",
    "italy",
    "morroco",
    "spain",
    "guinea",
    "tunisia",
    "eritrea",
    "palestine",
    "israel",
    "iran",
    "uzbekistan",
]

all_filtered_out_kwords = (
    syrian_location_specific_keywords
    + venezuela_specific_locations
    + nationalities_kwords
    + non_extracted_countries_kwords
)


countries_kwords = ["syria", "venezuela"]
augmented_country_kwords = countries_kwords + ["canada"]


def _generate_augmented_excerpt_nationalities(
    excerpt: str, original_nationality: str, one_new_nationality: str
):
    return _replace_kw(
        excerpt, clean_word=original_nationality, new_kw=one_new_nationality
    )


def _generate_augmented_excerpt_countries(
    excerpt: str, original_country: str, one_new_country: str
):
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
        if clean_word == original_country:
            changed_word = _replace_kw(
                one_word, clean_word=clean_word, new_kw=one_new_country
            )
        else:
            changed_word = one_word

        new_excerpt.append(f"{changed_word}{added_str_in_end}")

    return " ".join(new_excerpt)


def _create_country_df(df: pd.DataFrame):
    original_df = df.copy()
    original_df["country_context_falsing_kw"] = original_df.excerpt.apply(
        lambda x: [
            one_loc for one_loc in all_filtered_out_kwords if one_loc in x.lower()
        ]
    )
    original_df["country_keywords"] = original_df.excerpt.apply(
        lambda x: _extract_kwords(x, countries_kwords, exact_extraction=True)
    )

    original_df["country_kword_type"] = original_df[
        "country_keywords"
    ]  # because exact extraction

    return original_df


def _augment_country_df(df: pd.DataFrame) -> pd.DataFrame:
    final_df = pd.DataFrame()
    for one_original_country in countries_kwords:
        augmented_countries = [
            tmp_country
            for tmp_country in augmented_country_kwords
            if one_original_country != tmp_country
        ]
        df_one_kw = df[
            df.country_kword_type.apply(lambda x: x == [one_original_country])
        ].copy()
        for augmented_country in augmented_countries:
            augmented_df_one_country = df_one_kw.copy()
            augmented_df_one_country["country_kword_type"] = [
                [augmented_country] for _ in range(len(augmented_df_one_country))
            ]
            augmented_df_one_country["excerpt_type"] = "augmented"
            augmented_df_one_country["country_keywords"] = [
                [augmented_country] for _ in range(len(augmented_df_one_country))
            ]
            augmented_df_one_country[
                "excerpt"
            ] = augmented_df_one_country.excerpt.apply(
                lambda x: _generate_augmented_excerpt_countries(
                    x, one_original_country, augmented_country
                )
            )
            final_df = pd.concat([final_df, augmented_df_one_country])
    return final_df


def create_country_augmented_dataset(df: pd.DataFrame):
    # create augmented df
    final_df = _create_country_df(df)
    augmented_df = _augment_country_df(final_df)
    augmented_df = pd.concat([final_df, augmented_df])

    return augmented_df


def _get_mask_one_country(df: pd.DataFrame) -> pd.Series:
    mask_one_country = (
        (df.country_context_falsing_kw.apply(lambda x: len(x) == 0))
        & (df.country_kword_type.apply(lambda x: len(x) == 1))
        & (df.original_language == "en")
    )
    return mask_one_country


def _keep_one_country_df(df: pd.DataFrame) -> pd.DataFrame:
    mask_one_country = _get_mask_one_country(df)
    final_df = df.copy()[mask_one_country][
        [
            "entry_id",
            "excerpt",
            "country_keywords",
            "country_kword_type",
            "excerpt_type",
            "target",
            "original_language",
        ]
    ]

    return final_df


"""
def probing_nationality_country_original_df(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    task: str,
    scrape: bool,
):

    possible_tasks = ["nationalities", "countries"]
    assert task in possible_tasks, f"'task' argument must be in {possible_tasks}."
    if task == "nationalities":
        developing_kwords = developing_nationalities
        # developed_kword = developed_nationality
    else:
        developing_kwords = developing_countries
        # developed_kword = developed_country

    all_kwords = developing_kwords  # + [developed_kword]

    final_dfs = []
    for df in [df_train, df_val, df_test]:
        # filter out excerpt that mention other locations or nationalities....
        if scrape:
            original_df = _scrape(
                create_nationality_country_augmented_dataset(df, task)
            )
            original_df = original_df.rename(columns={"kword_type": "target"})
            original_df = original_df[["entry_id", "excerpt", "target"]]
            final_dfs.append(original_df)
        else:
            original_df = _create_nationality_df(df, task, all_kwords).rename(
                columns={"kword_type": "target"}
            )

            original_df = original_df[["entry_id", "excerpt", "target"]]
            final_dfs.append(original_df)

    return final_dfs"""
