import networkx as nx
from ast import literal_eval
from typing import List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer
from googletrans import Translator
import json

import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words())

############################### PREPROCESSING ########################################

affected_groups_list = [
    "Asylum Seekers",
    "Host",
    "IDP",
    "Migrants",
    "Refugees",
    "Returnees",
]


def flatten(t):
    """
    flatten a list of lists.
    """
    return [item for sublist in t for item in sublist]


def custom_eval(x):
    """
    custom evaluation function
    """
    if str(x) == "nan":
        return []
    if str(x) == "[None]":
        return []
    if type(x) == list:
        return x
    else:
        return literal_eval(x)


def get_number_of_clusters(n_entries: int):
    """
    get number of paragraphs we want depending on number of excerpts.
    """
    return min((n_entries // 10) + 1, 3)


def get_severity_score(item: List[str]):
    """
    change severity name to a severity score.
    """
    severity_mapper = {"Critical": 1, "Major": 0.75, "Of Concern": 0.5}

    if len(item) == 0:
        return 0.5
    severity_name = item[0]
    if severity_name in list(severity_mapper.keys()):
        return severity_mapper[severity_name]
    else:
        return 0.5


def get_reliability_score(item: List[str]):
    """
    change reliability name to a reliability score.
    """
    reliability_mapper = {
        "Completely Reliable": 1,
        "Usually reliable": 0.75,
        "Fairly Reliable": 0.5,
        "Not Usually Reliable": 0.25,
    }

    if len(item) == 0:
        return 0.5
    severity_name = item[0]
    if severity_name in list(reliability_mapper.keys()):
        return reliability_mapper[severity_name]
    else:
        return 0.5


def translate_sentence(excerpt: str):
    """
    function to translate excerpt to english.
    """
    translator = Translator()
    result = translator.translate(excerpt, dest="en")
    return result.text


def preprocess_excerpt(excerpt):
    """
    function to preprocess each excerpt
    - lower, remove punctuation
    - omit stop words
    """
    # lower and remove punctuation
    new_excerpt = re.sub(r"[^\w\s]", "", excerpt).lower()
    # omit stop words
    new_excerpt = " ".join(
        [word for word in new_excerpt.split(" ") if word not in stop_words]
    )
    return new_excerpt


def preprocesss_row(tags: List[str], n_tags: int):
    """
    put data in a standard way. When there are too many labels in one tag, it goes as "General Overview"
    """
    if len(tags) > n_tags:
        return ["General Overview"]
    else:
        return tags


def preprocess_df(df):
    """
    function to preprocess the original DataFrame. Works inplace.
    - apply literal evaluation
    - clean tags
    - get reliability scores / severity score
    - get number of sectors / secondary tags / affected groups
    """

    for col in [
        "sectors",
        "affected_groups",
        "specific_needs_groups",
        "age",
        "gender",
        "subpillars_1d",
        "subpillars_2d",
        "severity",
    ]:
        df[col] = df[col].apply(lambda x: list(set([item for item in custom_eval(x)])))

    df["pillars_2d"] = df.subpillars_2d.apply(
        lambda x: [item.split("->")[0] for item in x]
    )
    df["pillars_1d"] = df.subpillars_1d.apply(
        lambda x: [item.split("->")[0] for item in x]
    )
    df["affected_groups"] = df.affected_groups.apply(
        lambda x: [item for item in x if item not in ["None", "Others of Concern"]]
    )
    df["sectors"] = df.sectors.apply(lambda x: [item for item in x if item != "Cross"])
    df["n_sectors"] = df.sectors.apply(lambda x: len(x))
    df["severity_scores"] = df["severity"].apply(lambda x: get_severity_score(x))
    df["reliability_scores"] = df["reliability"].apply(
        lambda x: get_reliability_score(x)
    )
    df["secondary_tags"] = df.apply(
        lambda x: flatten(
            [f"{col}->{x[col]}" for col in ["specific_needs_groups", "age"]]
        ),
        axis=1,
    )
    df["n_secondary_tags"] = df.secondary_tags.apply(lambda x: len(x))
    df["n_affected_groups"] = df.affected_groups.apply(lambda x: len(x))


################################# GET SUMMARY ONE PART ##########################################3


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


def get_embeddings(texts):
    """
    get embeddings for each excerpt
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings_all_sentences = model.encode(texts)

    return np.array(embeddings_all_sentences)


def get_similarity_matrix(texts):
    """
    get similarity matrix between different excerpts
    """
    embeddings_all_sentences = get_embeddings(texts)

    similarity = cosine_similarity(embeddings_all_sentences, embeddings_all_sentences)
    return similarity


def t2t_generation(entries: str, english_returns: bool):
    """
    generate summary for set of entries
    """
    if english_returns:
        # en_entries = translate_sentence(entries)
        MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
        model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
        tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

        inputs = tokenizer([entries], return_tensors="pt")

        # Generate Summary
        summary_ids = model.generate(
            inputs["input_ids"], num_beams=4, num_beam_groups=2
        )
        summarized_entries = tokenizer.batch_decode(
            summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    else:
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        model = T5ForConditionalGeneration.from_pretrained("t5-large")

        changed_text = f"summarize: {entries}"
        input_ids = tokenizer(
            changed_text, return_tensors="pt", truncation=False
        ).input_ids
        outputs = model.generate(input_ids)
        summarized_entries = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summarized_entries


def get_summary_one_row(row, en_summary, n_clusters=1):
    """
    function used for getting summary for one task
    """

    def get_top_n_sentence_ids(graph):
        """
        get top 5 sentences based on based on reliability score / severity score / pagerank scores
        """
        scores_pagerank = nx.pagerank(graph)
        scores = {
            key: value * severity_scores[key] * reliability_scores[key]
            for key, value in scores_pagerank.items()
        }

        top_n_sentence_ids = np.argsort(np.array(list(scores.values())))[-5:]
        return top_n_sentence_ids

    def get_summary_one_cluster(cleaned_text):
        """
        get summary for each cluster
        1 - compute cosine similarity
        2 - build undirected graph based on similarity matrix between excerpts
        3 - get top n sentences
        4 - generate summary
        #"""
        cosine_similarity_matrix = get_similarity_matrix(cleaned_text)
        graph_one_lang = build_graph(cosine_similarity_matrix)
        top_n_sentence_ids = get_top_n_sentence_ids(graph_one_lang)
        ranked_sentence = " ".join(
            [original_excerpts[id_tmp] for id_tmp in (top_n_sentence_ids)]
        )
        summarized_entries = t2t_generation(ranked_sentence, en_summary)
        return summarized_entries

    if n_clusters < 1:
        AssertionError("number of clusters must be at least 1.")

    original_excerpts = row.excerpt
    severity_scores = row.severity_scores
    reliability_scores = row.reliability_scores

    # preprocess, clean excerpts
    cleaned_excerpts = np.array(
        [preprocess_excerpt(one_excerpt) for one_excerpt in original_excerpts]
    )

    if n_clusters == 1:
        summarized_entries = get_summary_one_cluster(cleaned_excerpts)
    else:
        summarized_entries = ""
        embeddings_all_sentences = get_embeddings(cleaned_excerpts)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
            embeddings_all_sentences
        )
        cluster_labels = kmeans.labels_
        for i in range(n_clusters):
            sentences_one_cluster = cleaned_excerpts[np.argwhere(cluster_labels == i)]
            summarized_entries += get_summary_one_cluster(sentences_one_cluster)

    return summarized_entries


def order_dict(x):
    """
    get the 'General Overview' tag as the first in report
    """
    if "General Overview" in list(x.keys()):
        first_dict = {"General Overview": x["General Overview"]}
        second_dict = {k: v for k, v in x.items() if k != "b"}
        y = {**first_dict, **second_dict}
        return y
    else:
        return x


def process_df_one_part(df: pd.DataFrame, col_name: str):
    """
    group df by col_name and get excerpts list for each different tag
    """
    df["tmp_tag_str"] = df[col_name].apply(str)

    grouped_df = df.groupby("tmp_tag_str", as_index=False)[
        ["excerpt", "severity_scores", "reliability_scores"]
    ].agg(lambda x: list(x))

    grouped_df["len"] = grouped_df["excerpt"].apply(lambda x: len(x))
    grouped_df = grouped_df[grouped_df.len >= 3]
    grouped_df["n_clusters"] = grouped_df["len"].apply(get_number_of_clusters)

    grouped_df.sort_values(by="len", ascending=False, inplace=False).drop(
        columns="tmp_tag_str", inplace=False
    )

    return grouped_df


def get_summary_one_part(df, col_name: str, en_summary, n_paragraphs=1):
    """
    main function to get the summary for one part
    example:
        - col_name: sectors
        - we get a paragraph for each different sector using the function 'get_summary_one_row'
    """
    preprocessed_df = df[df[col_name].apply(lambda x: len(x) > 0)].copy()
    if len(preprocessed_df) < 3:
        return {}
    else:
        preprocessed_df = process_df_one_part(df, col_name)

        final_returns = {}
        for index, row in preprocessed_df.iterrows():
            tag_tmp = row.tmp_tag_str
            summarized_text = get_summary_one_row(row, en_summary, n_paragraphs)
            final_returns[tag_tmp] = summarized_text

        return order_dict(final_returns)


###################################### SECTION 1 #####################################################


def get_report_context_crisis(df_one_project: pd.DataFrame, en_summary: bool):
    """
    section 1.1
    """
    df_context = df_one_project[
        df_one_project.pillars_1d.apply(lambda x: "Context" in x)
    ].copy()

    df_context["context_tags"] = df_context.subpillars_1d.apply(
        lambda x: preprocesss_row(
            [item.split("->")[1] for item in x if "Context" in item], 1
        )
    )

    return get_summary_one_part(df_context, "context_tags", en_summary)


def get_report_shocks_impacts(df_one_project, en_summary):
    """
    section 1.2
    """
    summary = {}
    impact_df = df_one_project[
        df_one_project.pillars_2d.apply(lambda x: "Impact" in x)
    ].copy()

    # Impact on people
    impact_people_df = impact_df[
        impact_df.subpillars_2d.apply(lambda x: "Impact->Impact On People" in x)
    ].copy()
    if len(impact_people_df) == 0:
        summary["Impact on people"] = {}
    else:
        impact_people_df["sectors"] = impact_people_df["sectors"].apply(
            lambda x: preprocesss_row(x, 1)
        )
        summary["Impact on people"] = get_summary_one_part(
            impact_people_df, "sectors", en_summary
        )

    # Impact on systems and services
    impact_systems_services_df = impact_df[
        impact_df.subpillars_2d.apply(
            lambda x: "Impact->Impact On System & Services" in x
        )
    ].copy()
    impact_systems_services_df["sectors"] = impact_systems_services_df["sectors"].apply(
        lambda x: preprocesss_row(x, 1)
    )
    summary["Impact on systems and services"] = get_summary_one_part(
        impact_systems_services_df, "sectors", en_summary
    )

    # Humanitarian Access, TODO: ADD NUMBERS HERE!!
    hum_access_df = df_one_project[
        df_one_project.pillars_1d.apply(lambda x: "Humanitarian Access" in x)
    ].copy()
    hum_access_df["non_number_hum_access"] = hum_access_df["subpillars_1d"].apply(
        lambda x: preprocesss_row(
            [item for item in x if item.split("->")[0] == "Humanitarian Access"], 1
        )
    )
    summary["Humanitarian Access"] = get_summary_one_part(
        hum_access_df, "non_number_hum_access", en_summary
    )

    return summary


def get_hum_report_one_affected_group(df: pd.DataFrame, tag: str, en_summary: bool):
    """
    one pop in section 1.4,
    """
    if tag == "Overall Tendencies":
        one_affected_group_df = df[df.n_affected_groups > 1]
    else:
        one_affected_group_df = df[df.affected_groups.apply(lambda x: [tag] == x)]

    one_affected_group_df[
        "non_num_hum_conds"
    ] = one_affected_group_df.subpillars_2d.apply(
        lambda x: [
            item
            for item in x
            if item
            in [
                "Humanitarian Conditions->Living Standards",
                "Humanitarian Conditions->Physical And Mental Well Being",
                "Humanitarian Conditions->Coping Mechanisms",
            ]
        ]
    )

    return get_summary_one_part(
        one_affected_group_df, "non_num_hum_conds", en_summary, 2
    )


def get_report_hum_conditions(df_one_project: pd.DataFrame, en_summary: bool):
    """
    section 1.4
    """
    df_all_hum_conditions = df_one_project[
        df_one_project.pillars_2d.apply(lambda x: "Humanitarian Conditions" in x)
    ].copy()
    summary = {}

    # General Findings
    summary["Overall Tendencies"] = get_hum_report_one_affected_group(
        df_all_hum_conditions, "Overall Tendencies", en_summary
    )

    # specific to each affected group
    for one_group in affected_groups_list:
        summary[one_group] = get_hum_report_one_affected_group(
            df_all_hum_conditions, one_group, en_summary
        )

    return summary


###################################### SECTION 3 #####################################################


def get_report_one_sector(one_project_df: pd.DataFrame, tag: str, en_summary: bool):
    """
    section 3 one sector
    """
    df_one_sector = one_project_df[
        one_project_df.sectors.apply(lambda x: tag in x)
    ].copy()

    summaries_one_sector = {}

    # secondary tagsaffected_groups
    summaries_one_sector["Most affected population groups"] = get_report_secondary_tags(
        df_one_sector, en_summary
    )

    # key trends
    summaries_one_sector["Key trends"] = get_report_hum_needs(df_one_sector, en_summary)

    # relation to other sectors
    summaries_one_sector[
        "Needs, severity and linkages with other sectors"
    ] = get_report_relation_other_sectors(df_one_sector, en_summary)

    return summaries_one_sector


def get_report_secondary_tags(df, en_summary):
    """
    get "Most affected population groups" in section 3
    """
    df_secondary_tags = df[
        (df.n_sectors == 1) & (df.secondary_tags.apply(lambda x: len(x) > 0))
    ].copy()

    df_secondary_tags["secondary_tags"] = df_secondary_tags["secondary_tags"].apply(
        lambda x: preprocesss_row(x, 1)
    )

    return get_summary_one_part(df_secondary_tags, "secondary_tags", en_summary)


def get_report_relation_other_sectors(df, en_summary):
    """
    section "Needs, severity and linkages with other sectors" in part 3
    """
    many_tags_df = df[(df.n_sectors > 1) & (df.n_secondary_tags != 1)].copy()
    many_tags_df["sectors"] = many_tags_df.sectors.apply(
        lambda x: preprocesss_row(x, 2)
    )
    # two_sectors_df = many_tags_df[many_tags_df.sectors.apply(lambda x: len(x)==2)]
    return get_summary_one_part(many_tags_df, "sectors", en_summary)


def get_report_hum_needs(df, en_summary):
    """
    get "key trends" in section 3
    """
    hum_conds_df = df[
        (df.pillars_2d.apply(lambda x: len(x) > 0))
        & (df.n_sectors == 1)
        & (df.n_secondary_tags != 1)
    ].copy()

    hum_conds_df["pillars_2d"] = hum_conds_df.pillars_2d.apply(
        lambda x: preprocesss_row(x, 1)
    )

    return get_summary_one_part(hum_conds_df, "pillars_2d", en_summary)


###################################### MAIN FUNCTION #####################################################


def get_report(
    full_df,
    project_id: int = None,
    en_summary: bool = True,
    save_summary: bool = True,
    use_sample: bool = False,
    summarize_part_one: bool = True,
    summarize_part_three: bool = True,
):
    """
    main function to get full report
    INPUTS:
        - full_df: input DataFrame
        - project_id: project id we work on
        - en_summary: whether or not we want the summary to be in english or in the original input language
        - save_summary: whether or not we want to save the summary
        - use_sample: if we want to use a sample of the input data (for test) or not
        - summarize_part_one: whether or not we want to summarize part 1
        - summarize_part_three: whether or not we want to summarize part 3

    OUTPUTS:
        - dict containing the summaries

    """
    if project_id is not None:
        df_one_project = full_df[full_df.project_id == project_id].copy()
    else:
        df_one_project = full_df
    if use_sample:
        df_one_project = df_one_project.sample(frac=0.2)
    if use_sample:
        df_one_project["excerpt"] = df_one_project["excerpt"].apply(translate_sentence)

    preprocess_df(df_one_project)

    final_summary = {}

    if summarize_part_one:
        # Part 1: Impact of the crisis and humanitarian conditions
        summaries_impact_hum_conditions = {}

        # generate report for part 1.1
        print("begin Context of the crisis")
        summaries_impact_hum_conditions[
            "Context of the crisis"
        ] = get_report_context_crisis(df_one_project, en_summary)

        # generate report for part 1.2
        print("begin Shocks and impact of the crisis")
        summaries_impact_hum_conditions[
            "Shocks and impact of the crisis"
        ] = get_report_shocks_impacts(df_one_project, en_summary)

        # generate report for part 1.4
        """print('begin Humanitarian conditions and severity of needs')
        summaries_impact_hum_conditions[
            'Humanitarian conditions and severity of needs'
            ] = get_report_hum_conditions(df_one_project, en_summary)"""

        final_summary[
            "Impact of the crisis and humanitarian conditions"
        ] = summaries_impact_hum_conditions

        if save_summary:
            with open(f"first_section_report.json", "w") as fp:
                json.dump(summaries_impact_hum_conditions, fp)

    if summarize_part_three:
        # Part 3: Sectoral analysis
        sectors_list = list(set(flatten(df_one_project.sectors)))

        summaries_sectors = {}
        for one_sector in sectors_list:
            print(f"begin {one_sector}")
            summaries_sectors[one_sector] = get_report_one_sector(
                df_one_project, tag=one_sector, en_summary=en_summary
            )

        final_summary["Sectoral Analysis"] = summaries_sectors

        if save_summary:
            with open(f"third_section_report.json", "w") as fp:
                json.dump(summaries_sectors, fp)

    if save_summary and summarize_part_three and summarize_part_one:
        with open(f"full_report.json", "w") as fp:
            json.dump(final_summary, fp)

    return final_summary
