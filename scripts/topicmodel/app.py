# https://github.com/MaartenGr/BERTopic/issues/392#issuecomment-1006333143

import json

import streamlit as st

import pandas as pd
import numpy as np
import hdbscan

from datetime import datetime
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from spacy.lang.es.stop_words import STOP_WORDS as es_stop
from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer
from umap import UMAP

from sklearn.feature_extraction.text import CountVectorizer

from hdbscan_expt import HDBSCAN_MODEL

from tag_options import (
    ngram_options,
    sector_options,
    subpillars2d_options,
    subpillars1d_options,
    project_options
)

from utils import preprocess, calc_coherence

# Wrap the texts to the allowed column width
pd.set_option("display.max_colwidth", -1)

st.set_page_config(layout="wide", page_title="Topic Modelling Env")


# @st.cache(allow_output_mutation=True)
def get_intertopic_dist_map(topic_model):
    return topic_model.visualize_topics()


# @st.cache(allow_output_mutation=True)
def get_topic_keyword_barcharts(topic_model, total_topics):
    return topic_model.visualize_barchart(
        top_n_topics=min(total_topics, 20),
        n_words=10,
        height=300,
        width=400)


# Stopwords
stop_words = list(fr_stop) + list(es_stop) + list(en_stop) + list(('nan',))


df = pd.read_csv(
    "st_embeddings.csv",
    usecols=[
        "entry_id", "excerpt", "sectors", "subpillars_2d",
        "subpillars_1d", "analysis_framework_id", "lead_id",
        "project_id", "embed"
    ],
    engine="c"
)


# Add projectname
df["project_id"] = df["project_id"].apply(str)

with open("project_mapping.json", "r") as f:
    proj_mapping = json.loads(f.read())


def get_project_name(pname):
    return proj_mapping[pname]


df["project_name"] = df.project_id.apply(get_project_name)

sectors, subpillars2d, subpillars1d, projectid = st.columns([1, 1, 1, 1])
sectors_dd_selection = sectors.selectbox('Select Sectors', sector_options)
subpillars2d_dd_selection = subpillars2d.selectbox(
    'Select Subpillars2D', subpillars2d_options
)
subpillars1d_dd_selection = subpillars1d.selectbox(
    'Select Subpillars1D', subpillars1d_options
)

projectids_lst = list(df.project_id.unique())

projectid_dd_selection = projectid.selectbox('Select Project Name', project_options)

df["sectors"] = df["sectors"].apply(eval)
df["subpillars_2d"] = df["subpillars_2d"].apply(eval)
df["subpillars_1d"] = df["subpillars_1d"].apply(eval)

if sectors_dd_selection != "None":
    mask_sector = df.sectors.apply(lambda x: sectors_dd_selection in x)
    df = df[mask_sector]
if subpillars2d_dd_selection != "None":
    mask_2dpillar = df.subpillars_2d.apply(lambda x: subpillars2d_dd_selection in x)
    df = df[mask_2dpillar]
if subpillars1d_dd_selection != "None":
    mask_1dpillar = df.subpillars_1d.apply(lambda x: subpillars1d_dd_selection in x)
    df = df[mask_1dpillar]
if projectid_dd_selection != "None":
    mask_projectid = df.project_name.apply(lambda x: projectid_dd_selection in x)
    df = df[mask_projectid]

st.dataframe(df)
st.info(f"Total records found: {len(df)}")

# Seed topics input
final_seed_topics = list()
seed_topics = st.text_area("Enter seed topics(separated by comma)")
for lst_of_keywords in seed_topics.split("\n"):
    lst_of_kw = [x.strip() for x in lst_of_keywords.split(",")]
    final_seed_topics.append(lst_of_kw)

st.info(final_seed_topics)

st.write("Setting Custom Parameters")
n_grams_col, _, _ = st.columns([1, 1, 1])

ngrams_mapper = {
    "Unigram": 1,
    "Bigram": 2,
    "Trigram": 3
}

n_grams_selection = n_grams_col.selectbox("Select n-gram", ngram_options)

excerpts = df["excerpt"]
excerpts_len = len(excerpts)

if st.button("Run Topic Modeling"):
    excerpts.reset_index(inplace=True, drop=True)
    excerpts = excerpts.apply(preprocess)

    # model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    # embedding_model = SentenceTransformer(model_name)
    # embeddings = embedding_model.encode(excerpts, show_progress_bar=True)
    df.embed = df.embed.apply(eval)
    embeddings = np.array(df.embed.tolist())

    if excerpts_len >= 25:
        start_time = datetime.now()

        umap_model = UMAP(
            n_neighbors=10,
            n_components=50,
            min_dist=0.00,
            metric='cosine',
            random_state=42
        )
        # Perform hdbscan expt for best params.
        hdb_expt = HDBSCAN_MODEL(embeddings)
        hdb_best_score = hdb_expt.calc_score()
        hdb_best_params = hdb_expt.get_best_params()

        st.write(f"HDBScan DBCV score: {hdb_best_score}")
        st.write(f"HDBSCAN Params: {hdb_best_params}")

        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=hdb_best_params["min_cluster_size"],
            metric=hdb_best_params["metric"],  # 'euclidean',
            cluster_selection_method=hdb_best_params["cluster_selection_method"],  # 'eom',
            prediction_data=True,
            min_samples=hdb_best_params["min_samples"]
        )

        vectorizer_model = CountVectorizer(
            ngram_range=(1, ngrams_mapper.get(n_grams_selection, 1)),
            stop_words=stop_words,
            # min_df=2
        )
        bertopic_params = dict(
            language="multilingual",
            low_memory=True,
            # embedding_model=embedding_model,
            umap_model=umap_model,
            verbose=True,
            calculate_probabilities=False,
            # min_topic_size=5,
            nr_topics="auto",
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model
        )
        if any(final_seed_topics):
            bertopic_params["seed_topic_list"] = final_seed_topics

        model = BERTopic(**bertopic_params)

        topics, probs = model.fit_transform(excerpts, embeddings=embeddings)

        # st.dataframe(topic_docs_df)
        # st.write(model.get_topic_info())
        topics_docs_df = pd.DataFrame({"topics": topics, "excerpts": excerpts})

        total_topics = topics_docs_df.topics.nunique()
        st.info(f"Total topics found: {total_topics}")

        fig1 = get_topic_keyword_barcharts(model, total_topics)
        st.write(fig1)

        # fig2 = get_intertopic_dist_map(model)
        # st.write(fig2)

        tabs = st.tabs([f"Topic {i-1}" for i in range(total_topics)])

        for topic_num, tab in enumerate(tabs):
            with tab:
                data = topics_docs_df[
                    topics_docs_df["topics"] == topic_num - 1
                ]
                st.info(f"Total documents: {len(data)}")
                st.dataframe(
                    data=data,
                    width=800
                )

        reduced_embeddings = UMAP(
            n_neighbors=10,
            n_components=2,
            min_dist=0.01,
            metric='cosine'
        ).fit_transform(embeddings)
        fig2 = model.visualize_documents(
            excerpts,
            reduced_embeddings=reduced_embeddings
        )
        st.write(fig2)

        coherence_score = calc_coherence(model, excerpts, topics)

        st.info(f"Coherence Score: {coherence_score}")

        end_time = datetime.now()
        time_diff = end_time - start_time
        st.info(f"Total time to run clustering: {time_diff.total_seconds():.2f} seconds")
    else:
        st.error(f"{excerpts_len} excerpts data is not enough.")
        raise Exception(f"{excerpts_len} excerpts data is not enough.")
