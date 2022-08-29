from typing import List
import networkx as nx
import re
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
stop_words = set(stopwords.words())


def flatten(t):
    """
    flatten a list of lists.
    """
    return [item for sublist in t for item in sublist]


def build_graph(cosine_similarity_matrix):
    """
    function to build graoh from similarity matrix
    """
    graph_one_lang = nx.Graph()
    matrix_shape = cosine_similarity_matrix.shape
    for i in range(matrix_shape[0]):
        for j in range(matrix_shape[1]):
            # do only once
            if i < j:
                sim = cosine_similarity_matrix[i, j]
                graph_one_lang.add_edge(i, j, weight=sim)
                graph_one_lang.add_edge(j, i, weight=sim)

    return graph_one_lang


def preprocess_one_sentence(sentence):
    """
    function to preprocess one_sentence:
        - lower and remove punctuation
        - remove stop words
        - stem and lemmatize
    """

    if type(sentence) is not str:
        sentence = str(sentence)

    new_words = []
    words = sentence.split()

    for word in words:

        # lower and remove punctuation
        new_word = re.sub(r"[^\w\s]", "", (word))

        # keep clean words and remove hyperlinks
        bool_word_not_empty_string = new_word != ""
        bool_word_not_stop_word = new_word.lower() not in stop_words

        if bool_word_not_empty_string and bool_word_not_stop_word:

            # lemmatize
            new_word = wordnet_lemmatizer.lemmatize(new_word, pos="v")

            # stem
            new_word = porter_stemmer.stem(new_word)

            new_words.append(new_word)

    return " ".join(new_words).rstrip().lstrip()


def preprocess_sentences(all_tweets: List[str]):
    """
    Clean list of sentences.
    """
    return [preprocess_one_sentence(one_tweet) for one_tweet in all_tweets]


def get_n_words(text):
    """
    Simple method to approximate the number of words in text.
    """
    return len(text.split(" "))
