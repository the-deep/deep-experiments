from io import StringIO
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
import re
from nostril import nonsense
from urllib.parse import urlparse
import pandas as pd
import streamlit as st

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

MIN_NUM_TOKENS = 5
MIN_WORD_LEN = 4

#TODO: @Stefano please use prediction API
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
    sentence = re.sub(r'[^\w' + keep + ']', '', sentence)
    return sentence


def page_to_sentences(page):
    sentences = sent_tokenize(page)
    sentences = [preprocess_sentence(sentence) for sentence in sentences]
    return sentences


def pdf_parser(fp):
    with st.spinner("Converting PDF to text.."):
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
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


uploaded_file = st.file_uploader(
    "Upload", type=["pdf"], accept_multiple_files=False, key=None, help=None)
if uploaded_file is not None:
    sentences = pdf_parser(uploaded_file)
    print(sentences)
    preds = list(map(predict_sector, sentences))

    col1, col2, col3 = st.beta_columns([10, 1, 3])
    col1.write("Excerpt")
    col2.text(" ")
    col3.text("Sector")
    st.markdown("""---""")
    for sentence, sector in zip(sentences, preds):
        col1, col2, col3 = st.beta_columns([10, 1, 3])
        col1.write(sentence)
        col2.text("g")
        col3.text(sector)
        st.markdown("""---""")
