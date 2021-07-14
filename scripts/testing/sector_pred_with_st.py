import os
import json
import streamlit.components.v1 as components
from pathlib import Path
import sys

sys.path.append(str((Path(sys.path[0])).parent.parent.parent))
sys.path.append(".")
from deep.constants import SECTORS

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
APP_NAME = "pl-example"
MIN_NUM_TOKENS = 5
MIN_WORD_LEN = 4
ID_TO_SECTOR = {i: sector for i, sector in enumerate(SECTORS)}


# Create a _RELEASE constant. We'll set this to False while we're developing
# the component, and True when we're ready to package and distribute it.
# (This is, of course, optional - there are innumerable ways to manage your
# release process.)
_RELEASE = False

# Declare a Streamlit component. `declare_component` returns a function
# that is used to create instances of the component. We're naming this
# function "_component_func", with an underscore prefix, because we don't want
# to expose it directly to users. Instead, we will create a custom wrapper
# function, below, that will serve as our component's public API.

# It's worth noting that this call to `declare_component` is the
# *only thing* you need to do to create the binding between Streamlit and
# your component frontend. Everything else we do in this file is simply a
# best practice.

if not _RELEASE:
    _component_func = components.declare_component(
        # We give the component a simple, descriptive name ("custom_table_component"
        # does not fit this bill, so please choose something better for your
        # own component :)
        "custom_table_component",
        # Pass `url` here to tell Streamlit that the component will be served
        # by the local dev server that you run via `npm run start`.
        # (This is useful while your component is in development.)
        url="http://localhost:3001",
    )
else:
    # When we're distributing a production version of the component, we'll
    # replace the `url` param with `path`, and point it to to the component's
    # build directory:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "online_table_component/frontend/build")
    _component_func = components.declare_component("custom_table_component", path=build_dir)


# Create a wrapper function for the component. This is an optional
# best practice - we could simply expose the component function returned by
# `declare_component` and call it done. The wrapper allows us to customize
# our component's API: we can pre-process its input args, post-process its
# output value, and add a docstring for users.
def custom_table_component(data, key=None):
    # Call through to our private component function. Arguments we pass here
    # will be sent to the frontend, where they'll be available in an "args"
    # dictionary.
    #
    # "default" is a special argument that specifies the initial return
    # value of the component before the user has interacted with it.
    component_value = _component_func(
        data=data, key=key, default=0, componentType="OnlineTableComponent"
    )

    # We could modify the value returned from the component if we wanted.
    # There's no need to do this in our simple example - but it's an option.
    return component_value


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


def predict_sector(sentences):
    return ["sec1" for sentence in sentences]
    test_data = pd.DataFrame({"excerpt": sentences})
    input_json = test_data.to_json(orient="split")
    predictions = query_endpoint(app_name=APP_NAME, input_json=input_json)
    predictions = [ID_TO_SECTOR[pred["0"]] for pred in predictions]
    return predictions


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


def send_feed_back(feedback_text, sentence, prediction, pdf_file_name):
    try:
        bucket_name = "deep-test-environment"
        s3_client = boto3.client("s3")
        model_identifier = "paraphrase-multilingual-mpnet-base-v2_secotrs-and-pillars"
        feedback_obj = {
            "feedback_text": feedback_text,
            "sentence": sentence,
            "prediction": prediction,
            "pdf_file_name": pdf_file_name,
            "model_identifier": model_identifier,
        }
        feedback_obj = json.dumps(feedback_obj).encode("utf-8")
        file_name_on_s3 = f"feedback/{time.ctime().replace(' ', '_')}.json"
        s3_client.put_object(Body=feedback_obj, Bucket=bucket_name, Key=file_name_on_s3)
        st.success("Feedback submitted successfully.")
    except Exception as e:
        st.warning(e.message)


uploaded_file = st.file_uploader(
    "Upload", type=["pdf"], accept_multiple_files=False, key=None, help=None
)
if uploaded_file is not None:
    sentences = pdf_parser(uploaded_file)
    preds = predict_sector(sentences)

    total_iteration = len(sentences)
    data = []
    for i in range(total_iteration):
        data.append({"sentence": sentences[i], "prediction": preds[i]})
    # cleaned_table = data.replace("", None)
    json_data = json.dumps(data)
    clicked_text = custom_table_component(json_data)
    file_name = uploaded_file.name
    if clicked_text:
        feedback_text = clicked_text.get("feedback", "")
        sentence = clicked_text.get("sentence", "")
        prediction = clicked_text.get("prediction", "")
        pdf_file_name = uploaded_file.name
        send_feed_back(feedback_text, sentence, prediction, pdf_file_name)
