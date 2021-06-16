import re
from nostril import nonsense

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

MIN_NUM_TOKENS = 5
MIN_WORD_LEN = 4

url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
url_regex = re.compile(url_regex)


def preprocess_sentence(sentence):
    tokens = sentence.split(" ")
    if len(tokens) < MIN_NUM_TOKENS:
        return ""
    sensible_token_count = 0
    for token in tokens:
        if len(token) > MIN_WORD_LEN or (len(token) > 7 and not nonsense(token)):
            sensible_token_count += 1
    if sensible_token_count < MIN_NUM_TOKENS:
        return ""
    sentence = " ".join(tokens)
    sentence = url_regex.sub("", sentence)
    keep = re.escape("/\\$.:,;-_()[]{}!'\"% ")
    sentence = re.sub(r"[^\w" + keep + "]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = sentence.strip()
    return sentence


def page_to_sentences(page):
    page = re.sub(r"\s+", " ", page)
    sentences = sent_tokenize(page)
    sentences = [preprocess_sentence(sentence) for sentence in sentences]
    return sentences
