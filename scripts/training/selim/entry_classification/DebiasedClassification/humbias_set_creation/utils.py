import pandas as pd
from typing import List, Set
from string import punctuation
from copy import copy
from functools import reduce
from IPython.display import display
from IPython.core.display import HTML
from ast import literal_eval

################################## GENERAL UTILS ##############################################


def _run_n_labels_sanity_check(
    items_list: List[List], n_expected_items: int, task: str = ""
):
    assert all(
        [len(one_sublist) == n_expected_items] for one_sublist in items_list
    ), f"issue for {task}"


def _custom_eval(x) -> List:
    if str(x) == "nan":
        return []
    if str(x) == "[None]":
        return []
    if type(x) == list:
        return x
    else:
        return literal_eval(x)


def _extract_kwords(
    excerpt: str, kwords: List[str], exact_extraction: bool = True
) -> List[str]:
    tot_extracted_kwords = []
    low_excerpt = excerpt.lower()

    excerpt_words = low_excerpt.split(" ")
    for one_word in excerpt_words:
        if "'s" in one_word:
            one_word_no_punctuation = copy(one_word).replace("'s", "")
        else:
            one_word_no_punctuation = copy(one_word)

        one_word_no_punctuation = one_word_no_punctuation.translate(
            str.maketrans("", "", punctuation)
        )
        if exact_extraction:
            extracted_kword_one_word = [
                one_kw for one_kw in kwords if one_word_no_punctuation == one_kw
            ]
        else:
            extracted_kword_one_word = [
                one_word_no_punctuation
                for one_kw in kwords
                if one_kw in one_word_no_punctuation
            ]
        tot_extracted_kwords.extend(extracted_kword_one_word)

    return list(set(tot_extracted_kwords))


def _replace_kw(original_kw: str, clean_word: str, new_kw: str):
    return (
        original_kw.replace(clean_word, new_kw)
        .replace(clean_word.capitalize(), new_kw.capitalize())
        .replace(clean_word.upper(), new_kw.upper())
    )


def _multisets_intersection(set_list: List[Set]):
    return list(set.intersection(*set_list))


def _get_token_ids_one_excerpt(tokenizer, max_len: int, excerpt_text: str):
    input_ids = tokenizer(
        excerpt_text,
        None,
        truncation=True,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        return_token_type_ids=True,
    )["input_ids"]
    return set(input_ids)


def _clean_biases_dataset(df_task, max_len, tokenizer, task_name: str):
    """
    delete badly augmented entries.
    for each entry_id:
        we have original entries and augmented entries.
        tokenize excerpts (original and augmented)
        if length of intersection of tokens between original entry and augmented entry is 0:
            means two entries were the same
            there is therefore a problem in the augmentation process.
    """
    problems_biases_df = pd.DataFrame()
    final_task_bias_df = pd.DataFrame()

    ids_one_task = df_task.entry_id.unique().tolist()
    for one_id in ids_one_task:
        df_one_id = df_task[df_task.entry_id == one_id]
        tokens_one_id = [
            _get_token_ids_one_excerpt(
                excerpt_text=one_excerpt, max_len=max_len, tokenizer=tokenizer
            )
            for one_excerpt in df_one_id.excerpt
        ]

        tokens_intersection_one_id = _multisets_intersection(tokens_one_id)
        # get length of tokens not in intersection
        not_in_intersection_token_lengths = [
            len(
                [
                    one_token
                    for one_token in one_excerpt_tokens
                    if one_token not in tokens_intersection_one_id
                ]
            )
            for one_excerpt_tokens in tokens_one_id
        ]
        lengths_product = reduce(
            (lambda x, y: x * y), not_in_intersection_token_lengths
        )
        if lengths_product == 0:  # at least one has a problem
            problems_biases_df = pd.concat([problems_biases_df, df_one_id])
        else:
            final_task_bias_df = pd.concat([final_task_bias_df, df_one_id])

    """if len(problems_biases_df) > 0:
        print(
            f"Warning! problem_bias_df is not null for {task_name}. wrong entry_ids are {problems_biases_df.entry_id.unique().tolist()}"
        )
        print(f"problem dataset for {task_name}")
        display(
            HTML(problems_biases_df.drop(columns=["target_classification"]).to_html())
        )"""

    return final_task_bias_df


def _create_scraped_excerpt(row: pd.Series, protected_attribute: str) -> str:
    original_excerpt = row["excerpt"]
    extracted_kwords = row[f"{protected_attribute}_keywords"]

    scraped_excerpt = copy(original_excerpt)
    for one_kw in extracted_kwords:
        scraped_excerpt = _replace_kw(scraped_excerpt, one_kw, "")

    return scraped_excerpt
