import json
import time
from ast import literal_eval
import boto3
import st_aggrid
import pandas as pd
import streamlit as st


table = pd.read_csv("entries_vs_sentences_tags.csv")
##
table.loc[table["Pillars"].isna(), "Pillars"] = "[]"
table["Pillars"] = table["Pillars"].apply(literal_eval)
table["Pillars"] = table["Pillars"].apply(lambda x: ", ".join(set(x)))
table.loc[table["Sectors"].isna(), "Sectors"] = "[]"
table["Sectors"] = table["Sectors"].apply(literal_eval)
table["Sectors"] = table["Sectors"].apply(lambda x: ", ".join(set(x)))
##
table.loc[table["Predicted Pillars"].isna(), "Predicted Pillars"] = "[]"
table["Predicted Pillars"] = table["Predicted Pillars"].apply(literal_eval)
table["Predicted Pillars"] = table["Predicted Pillars"].apply(
    lambda x: ", ".join(set(x)))
table.loc[table["Predicted Sectors"].isna(), "Predicted Sectors"] = "[]"
table["Predicted Sectors"] = table["Predicted Sectors"].apply(literal_eval)
table["Predicted Sectors"] = table["Predicted Sectors"].apply(
    lambda x: ", ".join(set(x)))
##
st.set_page_config(layout="wide")
projects = table["Project Title"].unique().tolist()
selected_project_title = st.selectbox("Select a Project", projects)
selected_project = table[table["Project Title"].eq(selected_project_title)]

af_title = selected_project["Analysis Framework Title"].iloc[0]
st.write(f"Analytical Framework: **{af_title}**")
lead_urls = selected_project["Lead URL"].unique().tolist()
selected_url = st.selectbox("Select a Lead:", lead_urls)
selected_lead = selected_project[selected_project["Lead URL"].eq(selected_url)]
selected_lead.drop(columns=["Lead URL", "Project Title",
                            "Analysis Framework Title"], inplace=True, axis=1)
st.write(f"[Selected Lead URL]({selected_url})")
st.markdown("""
<style>
    table tr:nth-child(even) {
        background-color: #f2f2f2;
    }

table {
        width: 100%;
        overflow-x: auto;
    }
table td:nth-child(1) {
    display: none
}
table th:nth-child(1) {
    display: none
}
}
</style>
""", unsafe_allow_html=True)
st.table(selected_lead)
feedback_text = st.text_input("Feedback: ", )


def send_feed_back(feedback_text, lead_url, project_title, af_title):
    bucket_name = "deep-test-environment"
    s3_client = boto3.client('s3')
    model_identifier = "paraphrase-multilingual-mpnet-base-v2_secotrs-and-pillars"
    feedback_obj = {
        "feedback_text": feedback_text,
        "lead_url": lead_url,
        "project_title": project_title,
        "af_title": af_title,
        "model_identifier": model_identifier
    }
    feedback_obj = json.dumps(feedback_obj).encode('utf-8')
    file_name_on_s3 = f"feedback/{time.ctime().replace(' ', '_')}.json"
    s3_client.put_object(
        Body=feedback_obj, Bucket=bucket_name, Key=file_name_on_s3)


clicked = st.button("Submit Feedback")
if clicked:
    send_feed_back(feedback_text, selected_url,
                   selected_project_title, af_title)
