# to run the app: streamlit run app.py

import ast
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid


def better_rending_excerpt(excerpt: str, words_per_line: int) -> str:
    words = excerpt.split(" ")
    # return " ".join(words[:words_per_line])
    n_words = len(words)
    n_lines = (n_words // words_per_line) + 2
    returned_excerpt = []
    for i in range(n_lines - 1):
        words_one_line = " ".join(words[i * words_per_line : (i + 1) * words_per_line])
        returned_excerpt.append(words_one_line)
    return returned_excerpt


st.set_page_config(page_title="Dataset", layout="wide")

# DATA HERE
d = pd.read_csv("male.csv")

# DUMMY COLUMN HERE
d["relevant"] = ["" for _ in range(len(d))]
# d["male_kw"] = d["male_kw"].apply(ast.literal_eval)
d = d[["entry_id", "excerpt", "relevant"]]

final_df = pd.DataFrame()
empty_row = pd.DataFrame([["", "", ""]], columns=d.columns)
ids = d.entry_id.unique().tolist()

for i in range(d.shape[0]):
    row = d.iloc[[i]]
    row["excerpt"] = row["excerpt"].apply(
        lambda x: better_rending_excerpt(x, words_per_line=15)
    )
    row = row.explode("excerpt")
    final_df = final_df.append(row)
    final_df = final_df.append(empty_row)


grid_options = {
    "columnDefs": [
        {
            "headerName": "excerpt",
            "field": "excerpt",
            "editable": False,
        },
        {
            "headerName": "entry_id",
            "field": "entry_id",
            "editable": False,
        },
        {
            "headerName": "relevant",
            "field": "relevant",
            "editable": True,
        },
    ],
}

grid_return = AgGrid(final_df, grid_options)
new_df = grid_return["data"]


@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")


csv = convert_df(new_df)
title = st.text_input("Filename")

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name=f"{title}.csv",
    mime="text/csv",
)
