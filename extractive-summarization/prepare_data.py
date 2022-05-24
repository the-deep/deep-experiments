import rust_utils

from glob import glob
from pathlib import Path
import json
import regex as re
from typing import List

from nnsplit import NNSplit
import spacy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass
from functools import partial
from transformers import HfArgumentParser
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
tqdm.pandas()


@dataclass
class Args:
    excerpts_csv_path: str
    lead_dirs: List[str]
    output_path: str
    n_subsample: int = None


class Sentencizer:
    def __init__(self):
        model_names = {
            "en": "en_core_web_sm",
            "fr": "fr_core_news_sm",
            "es": "es_core_news_sm",
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


def flatten(lst):
    return [x for sublist in lst for x in sublist]


def main(args):
    train_df = pd.read_csv(args.excerpts_csv_path)

    if args.n_subsample is not None:
        train_df = train_df.sample(n=args.n_subsample, random_state=1234)

    train_df["lead_id"] = train_df["lead_id"].astype(int)
    train_df["project_id"] = train_df["project_id"].astype(int)

    sentencizer = Sentencizer()

    full_texts_languages = {
        i: (group["lang"].value_counts().index[0] if "lang" in group else "en")
        for i, group in train_df.groupby(["lead_id", "project_id"])
    }
    full_texts = {}
    full_texts_sentences = {}
    bar = tqdm(total=len(full_texts_languages))

    for lead_dir in args.lead_dirs:
        for project_dir in Path(lead_dir).iterdir():
            if not project_dir.is_dir():
                continue

            for path in project_dir.iterdir():
                if not str(path).endswith(".txt"):
                    continue

                lead_id = int(path.name[: -len(".txt")])
                project_id = int(project_dir.name)

                if (lead_id, project_id) not in full_texts_languages:
                    continue

                text = "".join(
                    line
                    for line in open(path).readlines()
                    if not line.startswith("******")
                )

                full_texts[(lead_id, project_id)] = text
                full_texts_sentences[(lead_id, project_id)] = sentencizer(
                    text, full_texts_languages[(lead_id, project_id)]
                )
                bar.update(1)

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

    exact_matches = train_df.parallel_apply(exact_match, axis=1)

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

        return (excerpt, row["entry_id"])

    fuzzy_matches = train_df[~exact_matches].progress_apply(find_fuzzy_match, axis=1)

    def process_batch(batch, f):
        all_sentences = [full_texts_sentences[example["id"]] for example in batch]

        for sentences, example in zip(all_sentences, batch):
            sentence_indices = []

            # could use either the raw excerpts or the fuzzily matched excerpts for matching on sentence-level
            # it probably does not make a big difference
            for e, e_source in example["excerpts"]:
                for excerpt_sentence in sentencizer(e, example["language"]):
                    sentence_indices.append(
                        sorted(
                            [
                                {
                                    "index": i,
                                    "distance": rust_utils.levenshtein(
                                        s, excerpt_sentence
                                    ),
                                    "source": e_source,
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
                        "excerpts": [
                            {"text": e, "source": e_source}
                            for e, e_source in example["excerpts"]
                        ],
                        "raw_excerpts": example["raw_excerpts"],
                    }
                )
                + "\n"
            )

    def process_group(group, f):
        lead_id, project_id = group.name

        text = full_texts[(lead_id, project_id)]
        exact_group_matches = exact_matches[group.index]

        excerpts = group[exact_group_matches][["excerpt", "entry_id"]].values.tolist()
        excerpts.extend(
            fuzzy_matches[exact_group_matches[~exact_group_matches].index].dropna()
        )

        raw_excerpts = group["excerpt"].tolist()

        for e, _ in excerpts:
            assert e in text

        language = full_texts_languages[(lead_id, project_id)]

        process_batch(
            [
                {
                    "id": (int(lead_id), int(project_id)),
                    "text": text,
                    "language": language,
                    "excerpts": excerpts,
                    "raw_excerpts": raw_excerpts,
                }
            ],
            f,
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w") as f:
        train_df.groupby(["lead_id", "project_id"]).progress_apply(
            partial(process_group, f=f)
        )


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    main(args)
