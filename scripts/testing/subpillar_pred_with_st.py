from io import StringIO
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
import re
from nostril import nonsense
from urllib.parse import urlparse
import streamlit as st
import pandas as pd

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

MIN_NUM_TOKENS = 5
MIN_WORD_LEN = 4


# TODO: @Stefano please use prediction API
def predict_sector(sentence):
    return "Cross"


def preprocess_sentence(sentence):
    sentence = re.sub(r"\s+", " ", sentence)
    tokens = sentence.split(" ")
    if len(tokens) < MIN_NUM_TOKENS:
        return ""
    # remoe url
    tokens = [token for token in tokens if not urlparse(token).scheme]
    sensible_token_count = 0
    for token in tokens:
        if len(token) > MIN_WORD_LEN or (len(token) > 7 and not nonsense(token)):
            sensible_token_count += 1
    if sensible_token_count < MIN_NUM_TOKENS:
        return ""
    sentence = " ".join(tokens)
    keep = re.escape("/\\$.:,;-_()[]{}!'\"% ")
    sentence = re.sub(r"[^\w" + keep + "]", "", sentence)
    return sentence


def page_to_sentences(page):
    sentences = sent_tokenize(page)
    sentences = [preprocess_sentence(sentence) for sentence in sentences]
    return sentences


def pdf_parser(fp):
    with st.spinner("Converting PDF to text.."):
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        # Create a PDF interpreter object.
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # Process each page contained in the document.
        sentences = []
        for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)
            parsed_page = retstr.getvalue()
            sentences.extend(page_to_sentences(parsed_page))

    seen = set()
    seen_add = seen.add
    sentences = [s for s in sentences if not (s in seen or seen_add(s))]
    return sentences


DATA_PATH = "data/test_environment/fastai-5ep-english.pickle"
data = pd.read_pickle(DATA_PATH).sample(20)
index, sentences, preds, targets = data.index, data.excerpt, data.Predictions, data.Targets

st.set_page_config(layout="wide")
col0, col1, col2, col3, col4, col5, col6 = st.beta_columns([2, 1, 10, 1, 7, 1, 7])
col0.write("Index")
col1.text(" ")
col2.write("Excerpt")
col3.text(" ")
col4.text("Prediction")
col5.text(" ")
col6.text("Target")
st.markdown("""---""")
for ind, sentence, pred, target in zip(index, sentences, preds, targets):
    col0, col1, col2, col3, col4, col5, col6 = st.beta_columns([2, 1, 10, 1, 7, 1, 7])
    col0.write(ind)
    col1.text(" ")
    col2.write(sentence)
    col3.text(" ")
    col4.text("\n".join(pred))
    col5.text(" ")
    col6.text("\n".join(target))
    st.markdown("""---""")
