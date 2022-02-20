import rust_utils

from glob import glob
from pathlib import Path
import json
import regex as re

import spacy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

leads_df = pd.read_csv("../data/frameworks_data/data_v0.7.1/leads.csv")
train_df = pd.read_csv("../data/frameworks_data/data_v0.7.1/train_v0.7.1.csv")

lead_tuples = set(leads_df.apply(lambda row: (row["id"], row["project_id"]), axis=1))
assert len(lead_tuples) == len(leads_df)

for i, row in train_df.iterrows():
    lead_tuple = (row["lead_id"], row["project_id"])

    assert lead_tuple in lead_tuples

# leads = set(
#    glob("../data/frameworks_data/raw_data_excel_exports/dump/lead_previews/*.txt")
# )
leads = set(glob("../texts/*.txt"))

full_texts = {}

for i, row in tqdm(leads_df.iterrows()):
    lead_id = row["id"]
    project_id = row["project_id"]
    # name = f"../data/frameworks_data/raw_data_excel_exports/dump/lead_previews/leadid_{lead_id}_projectid_{project_id}_leadpreview.txt"
    # name of texts extracted with new tool
    name = str(
        Path("../texts") / Path(row["url"].rstrip("/").split("/")[-1]).with_suffix(".txt")
        if row["url"] is not np.nan
        else None
    )

    if name in leads:
        if (lead_id, project_id) not in full_texts:
            text = "\n".join(line for line in open(name) if not line.startswith("*********"))
            text = re.sub("\n+", "\n", text)
            full_texts[(lead_id, project_id)] = text

has_text_mask = train_df.apply(
    lambda row: (row["lead_id"], row["project_id"]) in full_texts,
    axis=1,
)

print(f"Dropping {(~has_text_mask).sum()} excerpts with no full text.")

train_df = train_df[has_text_mask]


def exact_match(row):
    lead_id = row["lead_id"]
    project_id = row["project_id"]

    return row["excerpt"] in full_texts[(lead_id, project_id)]


exact_matches = train_df.progress_apply(exact_match, axis=1)


def find_fuzzy_match(row):
    lead_id = row["lead_id"]
    project_id = row["project_id"]

    full_text = full_texts[(lead_id, project_id)]
    matches = sorted(rust_utils.levenshtein_search(row["excerpt"], full_text), key=lambda m: m[2])

    if len(matches) == 0:
        return None

    m = matches[0]
    excerpt = None

    for i in (0, -1, 1):
        for j in (0, 1, -1):
            try:
                excerpt = full_text.encode("utf-8")[m[0] + i : m[1] + j].decode("utf-8")
                break
            except UnicodeDecodeError:
                pass
        if excerpt is not None:
            break

    return excerpt


fuzzy_matches = train_df[~exact_matches].progress_apply(find_fuzzy_match, axis=1)


class Sentencizer:
    def __init__(self):
        model_names = {
            "en": "en_core_web_sm",
            "fr": "fr_core_news_sm",
            "es": "es_core_news_sm",
        }
        self.models = {}

        max_length = max(len(text) for text in full_texts.values())

        for lang, model_name in model_names.items():
            model = spacy.load(model_name, disable=["parser", "ner"])
            model.add_pipe("sentencizer")
            model.max_length = max_length

            self.models[lang] = model

    def __call__(self, text, language):
        if isinstance(text, str):
            nlp = self.models[language]
            return [str(x) for x in nlp(text).sents]
        else:
            text = np.array(text)
            language = np.array(language)

            sentences = [None for _ in range(len(text))]

            for lang, model in self.models.items():
                indices = np.where(language == lang)[0]

                docs = model.pipe([str(t) for t in text[indices]])
                for i, doc in zip(indices, docs):
                    sentences[i] = [str(x) for x in doc.sents]

            return sentences


sentencizer = Sentencizer()
batch = []
batch_size = 2


def process_batch(batch, f):
    all_sentences = sentencizer(
        [text for (text, _, _) in batch], [language for (_, _, language) in batch]
    )

    for sentences, (text, excerpt, language) in zip(all_sentences, batch):
        sentence_indices = []

        # could use either the raw excerpts or the fuzzily matched excerpts for matching on sentence-level
        # it probably does not make a big difference
        for e in excerpt:
            for excerpt_sentence in sentencizer(e, language):
                sentence_indices.append(
                    sorted(
                        [
                            {
                                "index": i,
                                "distance": rust_utils.levenshtein(s, excerpt_sentence),
                            }
                            for i, s in enumerate(sentences)
                        ],
                        key=lambda x: x["distance"],
                    )[0]
                )

        f.write(
            json.dumps(
                {
                    "text": text,
                    "excerpt": excerpt,
                    "sentences": sentences,
                    "sentence_indices": sentence_indices,
                }
            )
            + "\n"
        )


with open("data_new.json", "w") as f:
    for (lead_id, project_id), group in tqdm(train_df.groupby(["lead_id", "project_id"])):
        text = full_texts[(lead_id, project_id)]

        exact_group_matches = exact_matches[group.index]

        excerpt = group[exact_group_matches]["excerpt"].tolist()
        excerpt.extend(fuzzy_matches[exact_group_matches[~exact_group_matches].index].dropna())

        for e in excerpt:
            assert e in text

        language = group["lang"].value_counts().index[0]

        batch.append((text, excerpt, language))

        if len(batch) >= batch_size:
            process_batch(batch, f)
            batch = []

    process_batch(batch, f)
