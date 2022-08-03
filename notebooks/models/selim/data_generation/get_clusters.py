from typing import List

import re
import numpy as np

import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer

porter_stemmer = PorterStemmer()
stop_words = set(stopwords.words())

from sentence_transformers import SentenceTransformer

import hdbscan
import umap.umap_ as umap


def clean_one_sentence(sentence):
    """
    function to clean tweets:
    1) remove links
    2) remove users
    3) lower and remove punctuation
    4) stem and lemmatize if english language

    NB: This function contains many expressions taken from different sources.
    """

    if type(sentence) is not str:
        sentence = str(sentence)

    new_words = []
    words = sentence.split()

    for word in words:

        # lower and remove punctuation
        new_word = re.sub(r"[^\w\s]", "", (word))

        # keep clean words and remove hyperlinks
        word_not_nothing = new_word != ""
        word_not_stop_word = new_word.lower() not in stop_words

        if word_not_nothing and word_not_stop_word:

            # lemmatize
            new_word = wordnet_lemmatizer.lemmatize(new_word, pos="v")

            # stem
            new_word = porter_stemmer.stem(new_word)

            new_words.append(new_word)

    return " ".join(new_words).rstrip().lstrip()


def clean_excerpts(all_tweets: List[str]):
    return [clean_one_sentence(one_tweet) for one_tweet in all_tweets]


def get_embeddings(texts):
    """
    get all tweets embeddings, one embedding per tweet
    """
    model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
    return model.encode(texts)


def get_hdbscan_partitions(tweets: List[str]):
    """
    function to get HDBscan partitions: inspired from https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

    INPUT: List of preprocessed tweets
    OUTPUT: cluster label of each tweet

    1) get embeddings of tweets
    2) data reduction algorithm: UMAP
    3) HDBscan clustering
    """
    # Embeddings
    embeddings = get_embeddings(tweets)

    if embeddings.shape[0] > 100:
        embeddings = umap.UMAP(
            n_neighbors=7, n_components=15, metric="cosine"
        ).fit_transform(embeddings)

    # Hdbscan
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=10, metric="euclidean", cluster_selection_method="eom"
    ).fit(embeddings)

    return cluster.labels_


def get_clusters_sentences(
    original_excerpts: List[str],
):
    """
    Main function for clustering.
    INPUTS:
        -df: original DataFrame
        - clustering_method: method we use for clustering tweets: can be 'louvain' or 'hdbscan'.
        - language: one of ['en', 'fr, 'ar']
        - louvain_similarity_method: if we use the 'louvain' clustering method,
        choose one method to compute simialrity between 'tf-idf' and 'embeddings'.

    OUTPUTS:

    1) preprocess tweets
    2) get clustters
    3) get topics for tweets being in clusters
    """

    # preprocess tweets
    cleaned_excerpts = clean_excerpts(original_excerpts)
    cluster_ids = get_hdbscan_partitions(cleaned_excerpts)

    unique_labels = np.unique(cluster_ids)
    dict_grouped_excerpts = {one_label: [] for one_label in unique_labels}

    for i in range(len(cluster_ids)):
        dict_grouped_excerpts[cluster_ids[i]].append(original_excerpts[i])

    return dict_grouped_excerpts
