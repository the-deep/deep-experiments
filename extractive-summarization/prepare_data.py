import rust_utils

from glob import glob
from pathlib import Path
import json
import regex as re

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

valid_ids = [
    [39271, 2170],
    [46652, 2170],
    [40859, 1898],
    [38374, 1183],
    [42491, 1184],
    [44310, 2225],
    [43174, 2225],
    [56410, 2170],
    [50196, 2098],
    [51837, 2098],
    [56054, 2098],
    [38380, 1183],
    [49716, 2331],
    [45438, 2311],
    [55199, 2170],
    [45132, 2334],
    [21789, 729],
    [46445, 2098],
    [49648, 2225],
    [32339, 1620],
    [34498, 1620],
    [51267, 2336],
    [31209, 1900],
    [38714, 2098],
    [49217, 2311],
    [51963, 2225],
    [32916, 1184],
    [39097, 2170],
    [30918, 1183],
    [10948, 878],
    [51149, 2335],
    [37541, 1184],
    [56483, 2311],
    [55751, 2170],
    [49232, 2311],
    [31977, 1899],
    [32053, 1185],
    [39067, 2170],
    [56098, 2311],
    [45593, 2334],
    [49639, 2225],
    [51499, 2311],
    [29839, 1620],
    [45184, 2098],
    [22060, 788],
    [39908, 1388],
    [51121, 2311],
    [35868, 2028],
    [13739, 729],
    [51942, 2225],
    [50509, 2099],
    [53058, 2335],
    [45440, 2311],
    [47956, 2028],
    [37835, 2098],
    [31917, 1185],
    [26754, 1621],
    [14875, 729],
    [54424, 2333],
    [29946, 1231],
    [53066, 2335],
    [34493, 1620],
    [49740, 2332],
    [24414, 1620],
    [54799, 2099],
    [53320, 2098],
    [47706, 2311],
    [54878, 2332],
    [22134, 788],
    [29234, 1621],
    [13802, 729],
    [18965, 1388],
    [10826, 788],
    [31206, 1388],
    [16851, 788],
    [53700, 2170],
    [44565, 2331],
    [50199, 2170],
    [44369, 2099],
    [52935, 2336],
    [47461, 2332],
    [56655, 2225],
    [54290, 2332],
    [13636, 729],
    [53815, 2225],
    [34265, 1998],
    [9941, 729],
    [27409, 1621],
    [26134, 1620],
    [25249, 1621],
    [55448, 2170],
    [45353, 2099],
    [37822, 2098],
    [35873, 2028],
    [47389, 2311],
    [43819, 2330],
    [49642, 2225],
    [20372, 729],
    [14820, 788],
    [8997, 788],
    [9432, 788],
    [24146, 1620],
    [39028, 2171],
    [30866, 1621],
    [13635, 729],
    [9715, 730],
    [52766, 2225],
    [39946, 2171],
    [45780, 2333],
    [45556, 2311],
    [32725, 1620],
    [47523, 2225],
    [54620, 2028],
    [36134, 1185],
    [53814, 2335],
    [25173, 1621],
    [56637, 2225],
    [45678, 2333],
    [43309, 2332],
    [45188, 2098],
    [32504, 1620],
    [41311, 2028],
    [50168, 2170],
    [53036, 2334],
    [52556, 2099],
    [50890, 2466],
    [43491, 2225],
    [42879, 1183],
    [30210, 1184],
    [15114, 788],
    [16806, 1187],
    [52747, 2336],
    [8633, 788],
    [40702, 2098],
    [11927, 788],
    [48249, 2311],
    [19564, 788],
    [9238, 729],
    [28884, 1621],
    [39272, 2170],
    [40894, 1183],
    [38859, 1185],
    [43404, 2099],
    [31804, 1185],
    [47090, 2170],
    [55145, 2225],
    [26014, 1620],
    [13689, 729],
    [45191, 2098],
    [56379, 2311],
    [56484, 2311],
    [56280, 2098],
    [31908, 1185],
    [43988, 2311],
    [45239, 2225],
    [9246, 730],
    [23461, 1620],
    [37679, 1185],
    [53178, 2466],
    [45362, 2099],
    [31913, 1185],
    [44370, 2099],
    [23649, 1620],
    [34291, 1899],
    [36608, 1187],
    [8634, 788],
    [32917, 1184],
    [18445, 1183],
    [25332, 1620],
    [53618, 2170],
    [49365, 2311],
    [43375, 2028],
]


@dataclass
class Args:
    dataset: str  # 'new"
    leads_csv_path: str = "../data/frameworks_data/data_v0.7.1/leads.csv"
    excerpt_csv_path: str = "../data/frameworks_data/data_v0.7.1/train_v0.7.1.csv"
    lead_dirs: str = None
    out_path: str = None
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
    leads_df = pd.read_csv(args.leads_csv_path)
    train_df = pd.read_csv(args.excerpt_csv_path)

    if args.n_subsample is not None:
        train_df = train_df.sample(n=args.n_subsample, random_state=1234)

    train_df["lead_id"] = train_df["lead_id"].astype(int)
    train_df["project_id"] = train_df["project_id"].astype(int)

    lead_tuples = set(
        leads_df.apply(lambda row: (row["id"], row["project_id"]), axis=1)
    )
    assert len(lead_tuples) == len(leads_df)

    for i, row in train_df.iterrows():
        lead_tuple = (row["lead_id"], row["project_id"])

        assert lead_tuple in lead_tuples

    sentencizer = Sentencizer()

    full_texts_languages = {
        i: group["lang"].value_counts().index[0]
        for i, group in train_df.groupby(["lead_id", "project_id"])
    }
    full_texts = {}
    full_texts_sentences = {}
    bar = tqdm(total=len(full_texts_languages))

    for lead_dir in args.lead_dirs.split(","):
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
                        "train": example["train"],
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

        language = group["lang"].value_counts().index[0]

        process_batch(
            [
                {
                    "id": (int(lead_id), int(project_id)),
                    "text": text,
                    "language": language,
                    "excerpts": excerpts,
                    "raw_excerpts": raw_excerpts,
                    "train": [int(lead_id), int(project_id)] not in valid_ids,
                }
            ],
            f,
        )

    if args.out_path is None:
        out_path = Path(f"{args.dataset}.json")
    else:
        out_path = Path(args.out_path)

    out_path.parent.mkdir(exist_ok=True, parents=True)

    with open(out_path, "w") as f:
        train_df.groupby(["lead_id", "project_id"]).progress_apply(
            partial(process_group, f=f)
        )


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    main(args)
