from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from collections import defaultdict
import pandas as pd
from scipy.special import softmax
import os
from tqdm import tqdm
import argparse
from typing import List


def get_sentiment_one_task(tweets_list: List[str], model_name: str):
    """
    get sentiments column for each different task
    """
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tot_scores = defaultdict(list)

    # get sentiment score for each tweet
    for one_tweet in tqdm(tweets_list):

        encoded_input = tokenizer(one_tweet, return_tensors="pt")
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        for label_id, score in enumerate(scores):
            tot_scores[config.id2label[label_id]].append(score)

    return tot_scores


def get_sentiments_en(df: pd.DataFrame):
    """
    function to get sentiments when the language is english
    We work with 2 models here: 'sentiment' and 'emotion'.
    """

    data_df = df.copy()
    entries_list = data_df.excerpt.tolist()

    en_classification_tasks = ["sentiment", "emotion"]
    for task in en_classification_tasks:

        model_name = f"cardiffnlp/twitter-roberta-base-{task}"

        tot_scores = get_sentiment_one_task(entries_list, model_name)

        for label, scores in tot_scores.items():
            data_df[label] = scores

    return data_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", type=str)
    parser.add_argument(
        "--input_data_dir",
        type=str,  # default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,  # default=os.environ["SM_OUTPUT_DATA_DIR"]
    )

    args, _ = parser.parse_known_args()

    SUPPORTED_LANGUAGE_TYPES = [
        "en"
    ]  # change this when adding new languages to the pipeline.
    assert (
        args.language in SUPPORTED_LANGUAGE_TYPES
    ), f"'language' argument must be one of {SUPPORTED_LANGUAGE_TYPES}"

    data_df = pd.read_pickle(f"{args.input_data_dir}/data.pickle")[
        ["entry_id", "excerpt"]
    ].drop_duplicates(inplace=False)

    sentiments_df = get_sentiments_en(data_df)

    sentiments_df.to_csv(
        f"{args.output_data_dir}/sentiments_df.csv.gz",
        index=None,
        compression="gzip",
    )
