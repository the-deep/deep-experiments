import rust_utils

from glob import glob
from pathlib import Path
import json
import regex as re

import spacy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass

from transformers import HfArgumentParser

tqdm.pandas()


@dataclass
class Args:
    dataset: str  # 'new', 'old' or 'old_matching' or 'new_matching'
    leads_csv_path: str = "../data/frameworks_data/data_v0.7.1/leads.csv"
    excerpt_csv_path: str = "../data/frameworks_data/data_v0.7.1/train_v0.7.1.csv"
    leads_dir_old: str = None
    leads_dir_new: str = None
    out_path: str = None


def flatten(lst):
    return [x for sublist in lst for x in sublist]


def main(args):
    if args.leads_dir_old is None:
        leads_dir_old = Path(
            "../data/frameworks_data/raw_data_excel_exports/dump/lead_previews/"
        )
    else:
        leads_dir_old = args.leads_dir_old
    if args.leads_dir_new is None:
        leads_dir_new = Path("../texts/")
    else:
        leads_dir_new = args.leads_dir_new

    leads_df = pd.read_csv(args.leads_csv_path)
    train_df = pd.read_csv(args.excerpt_csv_path)

    lead_tuples = set(
        leads_df.apply(lambda row: (row["id"], row["project_id"]), axis=1)
    )
    assert len(lead_tuples) == len(leads_df)

    for i, row in train_df.iterrows():
        lead_tuple = (row["lead_id"], row["project_id"])

        assert lead_tuple in lead_tuples

    leads_old = (
        set(glob(str(leads_dir_old / "*.txt"))) if leads_dir_old is not None else set()
    )
    leads_new = (
        set(glob(str(leads_dir_new / "*.txt"))) if leads_dir_new is not None else set()
    )

    if "old" in args.dataset:
        leads = leads_old
    else:
        leads = leads_new

    full_texts = {}

    for i, row in tqdm(leads_df.iterrows()):
        lead_id = row["id"]
        project_id = row["project_id"]

        old_name = f"leadid_{lead_id}_projectid_{project_id}_leadpreview.txt"
        new_name = str(
            Path(row["url"].rstrip("/").split("/")[-1]).with_suffix(".txt")
            if row["url"] is not np.nan
            else None
        )

        old_path = str(leads_dir_old / old_name) if leads_dir_old is not None else None
        new_path = str(leads_dir_new / new_name) if leads_dir_new is not None else None

        if "old" in args.dataset:
            path = old_path
        else:
            path = new_path

        if args.dataset == "old_matching" and new_path not in leads_new:
            continue

        if args.dataset == "new_matching" and old_path not in leads_old:
            continue

        if path in leads:
            if (lead_id, project_id) not in full_texts:
                if "old" in args.dataset:
                    text = open(path).read()
                    text = re.sub("\n+", "\n", text)
                else:
                    text = "\n".join(
                        line for line in open(path) if not line.startswith("*********")
                    )

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
        matches = sorted(
            rust_utils.levenshtein_search(row["excerpt"], full_text), key=lambda m: m[2]
        )

        if len(matches) == 0:
            return None

        m = matches[0]
        excerpt = None

        for i in (0, -1, 1):
            for j in (0, 1, -1):
                try:
                    excerpt = full_text.encode("utf-8")[m[0] + i : m[1] + j].decode(
                        "utf-8"
                    )
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

        def sentencize(self, doc):
            sentences = []

            for sentence in doc.sents:
                start = sentence[0].idx
                end = (
                    sentence[-1].idx
                    + len(sentence[-1].text)
                    + len(sentence[-1].whitespace_)
                )

                sentences.append(doc.text[start:end])

            return sentences

        def __call__(self, text, language):
            if isinstance(text, str):
                nlp = self.models[language]
                return self.sentencize(nlp(text))
            else:
                text = np.array(text)
                language = np.array(language)

                sentences = [None for _ in range(len(text))]

                for lang, model in self.models.items():
                    indices = np.where(language == lang)[0]

                    docs = model.pipe([str(t) for t in text[indices]])
                    for i, doc in zip(indices, docs):
                        sentences[i] = self.sentencize(doc)

                return sentences

    sentencizer = Sentencizer()
    batch = []
    batch_size = 2

    def process_batch(batch, f):
        all_sentences = sentencizer(
            [example["text"] for example in batch],
            [example["language"] for example in batch],
        )

        for sentences, example in zip(all_sentences, batch):
            sentence_indices = []

            # could use either the raw excerpts or the fuzzily matched excerpts for matching on sentence-level
            # it probably does not make a big difference
            for e in example["excerpts"]:
                for excerpt_sentence in sentencizer(e, example["language"]):
                    sentence_indices.append(
                        sorted(
                            [
                                {
                                    "index": i,
                                    "distance": rust_utils.levenshtein(
                                        s, excerpt_sentence
                                    ),
                                }
                                for i, s in enumerate(sentences)
                            ],
                            key=lambda x: x["distance"],
                        )[0]
                    )

            f.write(
                json.dumps(
                    {
                        "id": example["id"],
                        "text": example["text"],
                        "sentences": sentences,
                        "excerpt_sentence_indices": sentence_indices,
                        "excerpts": example["excerpts"],
                        "raw_excerpts": example["raw_excerpts"],
                    }
                )
                + "\n"
            )

    if args.out_path == None:
        out_path = Path(f"{args.dataset}.json")
    else:
        out_path = Path(args.out_path)

    out_path.parent.mkdir(exist_ok=True, parents=True)

    with open(out_path, "w") as f:
        for (lead_id, project_id), group in tqdm(
            train_df.groupby(["lead_id", "project_id"])
        ):
            text = full_texts[(lead_id, project_id)]
            exact_group_matches = exact_matches[group.index]

            excerpts = group[exact_group_matches]["excerpt"].tolist()
            excerpts.extend(
                fuzzy_matches[exact_group_matches[~exact_group_matches].index].dropna()
            )

            raw_excerpts = group["excerpt"].tolist()

            for e in excerpts:
                assert e in text

            language = group["lang"].value_counts().index[0]

            batch.append(
                {
                    "id": (int(lead_id), int(project_id)),
                    "text": text,
                    "language": language,
                    "excerpts": excerpts,
                    "raw_excerpts": raw_excerpts,
                }
            )

            if len(batch) >= batch_size:
                process_batch(batch, f)
                batch = []

        process_batch(batch, f)


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    main(args)
