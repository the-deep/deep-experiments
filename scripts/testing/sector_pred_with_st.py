from pathlib import Path
import sys

sys.path.append(str((Path(sys.path[0])).parent.parent.parent))
sys.path.append(".")
from deep.constants import SECTORS

import json
import time
from io import StringIO
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
import pandas as pd
import boto3
import re
from nostril import nonsense
from urllib.parse import urlparse
import streamlit as st

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

RELEASE = False
APP_NAME = "pl-example"
MIN_NUM_TOKENS = 5
MIN_WORD_LEN = 4
ID_TO_SECTOR = {i: sector for i, sector in enumerate(SECTORS)}


def query_endpoint(app_name, input_json):
    client = boto3.session.Session().client("sagemaker-runtime", "us-east-1")

    response = client.invoke_endpoint(
        EndpointName=app_name,
        Body=input_json,
        ContentType="application/json; format=pandas-split",
    )
    preds = response["Body"].read().decode("ascii")
    preds = json.loads(preds)
    print("Received response: {}".format(preds))
    return preds

@st.cache
def predict_sector(sentences):
    if RELEASE:
        test_data = pd.DataFrame({"excerpt": sentences})
        input_json = test_data.to_json(orient="split")
        predictions = query_endpoint(app_name=APP_NAME, input_json=input_json)
        predictions = [ID_TO_SECTOR[pred["0"]] for pred in predictions]
        return predictions
    else:
        for _ in range(3):
            time.sleep(1)
        return ["Random"]*len(sentences)


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

@st.cache
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


uploaded_file = st.file_uploader(
    "Upload", type=["pdf"], accept_multiple_files=False, key="None", help=None
)
if uploaded_file is not None:
    if "fname" in st.session_state and st.session_state.fname == uploaded_file.name:
        sentences = st.session_state.sentences
        preds = st.session_state.preds
    else:
        sentences = pdf_parser(uploaded_file)
        preds = predict_sector(sentences)

    col1, col2, col3 = st.beta_columns([10, 1, 3])
    col1.write("Excerpt")
    col2.text(" ")
    col3.text("Sector")
    st.markdown("""---""")
    for sentence, sector in zip(sentences, preds):
        col1, col2, col3 = st.beta_columns([10, 1, 3])
        col1.write(sentence)
        col2.text(" ")
        col3.text(sector)
        st.markdown("""---""")
