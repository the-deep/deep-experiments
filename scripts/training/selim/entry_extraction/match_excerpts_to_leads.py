from glob import glob
from pathlib import Path
import json
import regex as re
from typing import List, Dict
import os
from nnsplit import NNSplit
import spacy
from nltk.tokenize import sent_tokenize
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass
import string
from ast import literal_eval
from copy import copy
import random
from tqdm import tqdm

tqdm.pandas()

languages = ["en", "fr", "es", "pt"]


@dataclass
class Args:
    excerpts_csv_path: str
    lead_dirs: List[str]
    output_path: str
    n_subsample: int = None


"""class Sentencizer:
    def __init__(self):
        model_names = {
            "en": "en_core_web_sm",
            "fr": "fr_core_news_sm",
            "es": "es_core_news_sm",
            "pt": "pt_core_news_sm",
        }
        self.models = {}
        self.sub_models = {}

        for lang, model_name in model_names.items():
            model = spacy.load(model_name, disable=["parser", "ner"])
            model.add_pipe("sentencizer")

            self.models[lang] = model
            try:
                self.sub_models[lang] = NNSplit.load(lang)
            except:
                pass

    def sub_sentencize(self, text, lang):
        if lang in self.sub_models:
            return [str(x) for x in self.sub_models[lang].split([text])[0]]
        else:
            return [str(text)]

    def sentencize(self, doc, lang):
        sentences = []

        for sentence in doc.sents:
            start = sentence[0].idx
            end = (
                sentence[-1].idx
                + len(sentence[-1].text)
                + len(sentence[-1].whitespace_)
            )

            text = doc.text[start:end]

            index = 0
            for match in re.finditer("\n+", text):
                sentences.extend(self.sub_sentencize(text[index : match.end()], lang))
                index = match.end()

            if index != len(text):
                sentences.extend(self.sub_sentencize(text[index:], lang))

        return sentences

    def __call__(self, text, language):
        for model in self.models.values():
            model.max_length = max(model.max_length, len(text))

        nlp = self.models[language]
        return self.sentencize(nlp(text), language)
"""


def flatten(lst):
    return [x for sublist in lst for x in sublist]


def longest_str_intersection(a: str, b: str):

    # identify all possible character sequences from str a
    seqs = []
    for pos1 in range(len(a)):
        for pos2 in range(len(a)):
            seqs.append(a[pos1 : pos2 + 1])

    # remove empty sequences
    seqs = [seq for seq in seqs if seq != ""]

    # find segments in str b
    max_len_match = 0
    max_match_sequence = ""
    for seq in seqs:
        if seq in b:
            if len(seq) > max_len_match:
                max_len_match = len(seq)
                max_match_sequence = seq

    return max_match_sequence


def _longest_common_subsequence(X: List, Y: List):
    # from here https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]


def _get_tagname_to_id(target) -> Dict[str, int]:
    """
    Assign id to each tag
    """
    tag_set = set()
    for tags_i in target:
        tag_set.add(tags_i)
    tagname_to_tagid = {tag: i for i, tag in enumerate(list(sorted(tag_set)))}
    return tagname_to_tagid


def _clean_one_entry(entry: str) -> List[str]:
    cleaned_entry = (
        str(entry).lower().translate(str.maketrans("", "", string.punctuation))
    )
    return cleaned_entry.split(" ")


def _select_relevant_tags(entry: str) -> List[str]:
    final_entry = literal_eval(copy(entry))
    cleaned_entry = ["is_relevant"]
    for one_tag in final_entry:
        if (
            "first_level_tags" in one_tag
            and "first_level_tags->sectors->Cross" != one_tag
        ):
            cleaned_entry.append(one_tag)
        """if "secondary_tags" in one_tag and all(
            [kw not in one_tag for kw in ["severity", "specific_needs_groups"]]
        ):
            cleaned_entry.append(one_tag)"""
    return cleaned_entry


def create_excerpts_data_version(
    excerpts_csv_path: str, raw_leads_dir: str, n_subsample: int = None
):

    train_df = pd.read_csv(excerpts_csv_path)
    train_df["nlp_tags"] = [
        ["is_relevant"] for _ in train_df.index
    ]  # train_df["nlp_tags"].apply(_select_relevant_tags)

    # sentencizer = Sentencizer()
    train_df["original_language"] = train_df["original_language"].apply(
        lambda x: x if x in languages else "en"
    )
    train_df["excerpt"] = train_df.apply(lambda x: x[x["original_language"]], axis=1)

    train_df["lead_id"] = train_df["lead_id"].astype(int)
    train_df["project_id"] = train_df["project_id"].astype(int)

    full_text = train_df.groupby("lead_id").agg(
        {
            "entry_id": lambda x: list(x),
            "excerpt": lambda x: list(x),
            "nlp_tags": lambda x: list(x),
            # "original_language": lambda x: Counter(list(x)).most_common(1)[0][0],
        }
    )

    # if subsampling, no need to run for everything, just a sample big enough to contain the subsample we will keep in the end.
    if n_subsample is not None:
        full_text = full_text.sample(n=n_subsample * 3, random_state=1234)

    full_text["n_excerpts"] = full_text.entry_id.apply(lambda x: len(x))

    raw_data = []

    projects_dirs = [id for id in os.listdir(raw_leads_dir) if id.isdigit()]
    all_leads = [
        {"project_id": project_id, "lead_id": lead_id}
        for project_id in projects_dirs
        for lead_id in os.listdir(os.path.join(raw_leads_dir, project_id))
    ]

    all_lead_ids = set(full_text.index.tolist())

    for one_lead_info in tqdm(all_leads):
        lead_path = one_lead_info["lead_id"]
        project_id = one_lead_info["project_id"]
        one_lead_id = int(lead_path.replace(".json", ""))

        if one_lead_id in all_lead_ids:

            with open(os.path.join(raw_leads_dir, project_id, lead_path)) as f:
                text = json.load(f)

            text = " ".join(flatten(text))

            lead_excerpts = full_text.loc[one_lead_id]

            lead_sentences = [
                sent for sent in sent_tokenize(text) if len(str(sent)) > 5
            ]
            if len(lead_sentences) > 3:

                clean_excerpts = [
                    _clean_one_entry(one_excerpt)
                    for one_excerpt in lead_excerpts.excerpt
                ]
                entry_ids = lead_excerpts.entry_id
                excerpts_tags = lead_excerpts.nlp_tags

                matched_sentences = []
                for sent_id, sentence in enumerate(lead_sentences):

                    clean_sentence = _clean_one_entry(
                        sentence
                    )  # lower and delete punctuation
                    n_words_sentence = len(clean_sentence)
                    max_n_words_difference = n_words_sentence // 5

                    if n_words_sentence > 2:
                        final_source = []
                        final_tags = []
                        final_distance = float("inf")

                        for one_entry_id, one_excerpt, one_tags in zip(
                            entry_ids, clean_excerpts, excerpts_tags
                        ):
                            # intersection_length = _longest_common_subsequence(
                            #    clean_sentence, one_excerpt
                            # )
                            intersection_length = len(
                                set(one_excerpt).intersection(set(clean_sentence))
                            )

                            distance_one_excerpt = (
                                n_words_sentence - intersection_length
                            )
                            if distance_one_excerpt == final_distance:
                                final_tags = list(
                                    set(final_tags).intersection(set(one_tags))
                                )
                                final_source.append(one_entry_id)

                            if distance_one_excerpt < final_distance:
                                final_distance = distance_one_excerpt
                                final_tags = one_tags
                                final_source = [one_entry_id]

                        if final_distance <= max_n_words_difference:
                            match_one_sentence = {
                                "index": sent_id,
                                "distance": final_distance,
                                "tags": final_tags,
                                "source": final_source,
                            }
                            matched_sentences.append(match_one_sentence)

                n_relevant_sentences = len(matched_sentences)
                n_total_sentences = len(lead_sentences)

                ratio_relevant_sentences = n_relevant_sentences / n_total_sentences

                if ratio_relevant_sentences > 0.1 and ratio_relevant_sentences < 0.7:
                    output_one_lead = {
                        "id": {"lead_id": one_lead_id, "project_id": int(project_id)},
                        "sentences": lead_sentences,
                        "excerpt_sentence_indices": matched_sentences,
                    }

                    raw_data.append(output_one_lead)

    if n_subsample is not None:
        raw_data = random.sample(raw_data, min(len(raw_data), n_subsample))

    tagname_to_id = _get_tagname_to_id(list(set(flatten(train_df["nlp_tags"]))))

    final_saved_data = {"data": raw_data, "tagname_to_tagid": tagname_to_id}

    dict_name = "entry_extraction_dict"
    df_name = "entry_extraction_df"
    if n_subsample is not None:
        dict_name = f"{dict_name}_{n_subsample}"
        df_name = f"{df_name}_{n_subsample}"

    with open(f"{dict_name}.json", "w") as fp:
        json.dump(final_saved_data, fp)

    output_as_df = pd.DataFrame(
        [
            [
                final_saved_data["data"],
                final_saved_data["tagname_to_tagid"],
            ]
        ],
        columns=["data", "tagname_to_tagid"],
    )
    output_as_df.to_csv(f"{df_name}.csv", index=None)
