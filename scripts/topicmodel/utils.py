import re
import numpy as np
import pandas as pd
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel


def preprocess(text):
    text = np.str_(text)
    text = text.replace('\\n', '')
    text = text.replace('\\', '')
    text = re.sub(r"\[(.*?)\]", '', text)  # removes [this one]
    url_pattern = (
        "((https?|ftp|smtp)://)?(www.)?[a-z0-9]+\\.[a-z]+(/[a-zA-Z0-9#]+/?)*"
    )
    text = re.sub(
        url_pattern,
        '',
        text
    )  # remove urls
    # text = re.sub('\'','',text)
    # text = re.sub(r'\d+', ' __number__ ', text) #replaces numbers
    # text = re.sub('\W', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.replace('\t', '')
    text = text.replace('\n', '')

    return text


def calc_coherence(model, data, topics):
    documents = pd.DataFrame({"Document": data,
                              "ID": range(len(data)),
                              "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    # words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [
        [words for words, _ in model.get_topic(topic)]
        for topic in range(len(set(topics)) - 1)]

    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words,
                                     texts=tokens,
                                     corpus=corpus,
                                     dictionary=dictionary,
                                     coherence='c_v')
    return coherence_model.get_coherence()
