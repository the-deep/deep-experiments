import networkx as nx
from ast import literal_eval
from typing import List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer
from get_clusters import get_clusters_sentences

# from googletrans import Translator
import json
import operator
import re
from nltk.corpus import stopwords
import nltk

stop_words = set(stopwords.words())

############################### PREPROCESSING ########################################

affected_groups_list = [
    "Asylum Seekers",
    "Host",
    "IDP",
    "Migrants",
    "Refugees",
    "Returnees",
    "Stateless",
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
    return min((n_entries // 30) + 1, 2)


def get_severity_score(item: List[str]):
    """
    change severity name to a severity score.
    """
    severity_mapper = {"critical": 1, "major": 0.75, "of concern": 0.5}

    if len(item) == 0:
        return 0.5
    severity_name = item[0]
    if severity_name.lower() in list(severity_mapper.keys()):
        return severity_mapper[severity_name]
    else:
        return 0.25


def get_numbers_score(item: List[str]):
    """
    if there is a relevant number in excerpt, give it more weight
    """
    n_number_subpillars = len(
        [one_subpillar for one_subpillar in item if "number" in one_subpillar.lower()]
    )
    if n_number_subpillars > 1:
        return 1
    elif n_number_subpillars == 1:
        return 0.75
    else:
        return 0.5


def clean_characters(text: str):
    # clean for latex characters
    latex_text = text.replace("%", "\%").replace("$", "\$")

    # strip punctuation
    latex_text = re.sub(r'\s([?.!"](?:\s|$))', r"\1", latex_text)

    return latex_text


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


def get_reliability_score(item: List[str]):
    """
    change reliability name to a reliability score.
    """
    reliability_mapper = {
        "completely reliable": 1,
        "usually reliable": 0.75,
        "fairly reliable": 0.5,
        "not usually reliable": 0.25,
    }

    if len(item) == 0:
        return 0.5
    reliability_name = item[0]
    if reliability_name.lower() in list(reliability_mapper.keys()):
        return reliability_mapper[reliability_name]
    else:
        return 0.5


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
    df["subpillars"] = df.apply(lambda x: x.subpillars_1d + x.subpillars_2d, axis=1)
    df["affected_groups"] = df.affected_groups.apply(
        lambda x: [item for item in x if item not in ["None", "Others of Concern"]]
    )
    df["sectors"] = df.sectors.apply(lambda x: [item for item in x if item != "Cross"])
    df["n_sectors"] = df.sectors.apply(lambda x: len(x))

    # scores used as weights later
    df["severity_scores"] = df["severity"].apply(get_severity_score)
    df["reliability_scores"] = df["reliability"].apply(get_reliability_score)
    df["present_numbers_score"] = df["subpillars"].apply(get_numbers_score)

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
    model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
    embeddings_all_sentences = model.encode(texts)

    return np.array(embeddings_all_sentences)


def get_similarity_matrix(embeddings_all_sentences):
    """
    get similarity matrix between different excerpts
    """

    similarity = cosine_similarity(embeddings_all_sentences, embeddings_all_sentences)
    return similarity


def t2t_generation(entries: str, english_returns: bool):
    """
    generate summary for set of entries
    """
    # if english_returns:

    MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

    inputs = tokenizer([entries], max_length=1024, return_tensors="pt", truncation=True)

    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, num_beam_groups=2)
    summarized_entries = tokenizer.batch_decode(
        summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return summarized_entries


def get_top_n_sentence_ids(
    graph,
    n_entries,
    severity_scores=None,
    reliability_scores=None,
    present_numbers_score=None,
):
    """
    get sentences scores based on based on reliability score / severity score / pagerank scores
    """
    try:
        scores_pagerank = nx.pagerank(graph)
        scores = {
            key: value
            # * severity_scores[key]
            # * reliability_scores[key]
            # * present_numbers_score[key]
            for key, value in scores_pagerank.items()
        }
        return np.array(list(scores.values()))

    except Exception as e:
        return np.array([1 for _ in range(n_entries)])


def get_summary_one_cluster(
    original_excerpts: List[str],
    severity_scores=None,
    reliability_scores=None,
    present_numbers_score=None,
) -> str:
    """
    get summary for each cluster
    1 - compute cosine similarity
    2 - build undirected graph based on similarity matrix between excerpts
    3 - get top n sentences
    4 - generate summary
    """
    cleaned_text = np.array(
        [preprocess_excerpt(one_excerpt) for one_excerpt in original_excerpts]
    )

    embeddings_all_sentences = get_embeddings(cleaned_text)
    cosine_similarity_matrix = get_similarity_matrix(embeddings_all_sentences)
    graph_one_lang = build_graph(cosine_similarity_matrix)
    scores = get_top_n_sentence_ids(
        graph_one_lang,
        len(cleaned_text),
        severity_scores,
        reliability_scores,
        present_numbers_score,
    )

    n_max_kept_sentences = 15

    top_n_sentence_ids = np.argsort(scores)[::-1][:n_max_kept_sentences]
    ranked_sentence = [original_excerpts[id_tmp] for id_tmp in (top_n_sentence_ids)]

    """top_n_sentences_token_lengths = [
        len(nltk.word_tokenize(sent)) for sent in ranked_sentence
    ]
    tot_tokens_len = 0
    final_used_sentences = []
    for i in range(n_max_kept_sentences):
        tot_tokens_len += top_n_sentences_token_lengths[i]
        if tot_tokens_len < 600:
            final_used_sentences.append(ranked_sentence[i])"""

    summarized_entries = clean_characters(
        t2t_generation(" ".join(ranked_sentence), english_returns=True)
    )

    return summarized_entries


def get_summary_one_row(row, en_summary=True, n_min_entries=10):
    """
    function used for getting summary for one task
    TODO: Embedding + preprocessing are being done twice
    TODO: reformat this part
    """
    if type(row) is list:
        original_excerpts = row
    else:
        original_excerpts = list(set(row.excerpt))

    # severity_scores = row.severity_scores
    # reliability_scores = row.reliability_scores
    # present_numbers_score = row.present_numbers_score
    # entry_ids = row.entry_id

    dict_clustered_excerpts = get_clusters_sentences(original_excerpts)

    n_clusters = len(dict_clustered_excerpts)

    if n_clusters == 1:
        summarized_entries = get_summary_one_cluster(
            original_excerpts,
            severity_scores=None,
            reliability_scores=None,
            present_numbers_score=None,
        )

    else:
        if n_clusters > 5:
            # keep only five longest clusters (which are different from -1)
            cluster_lengths = {
                k: len(v) for k, v in dict_clustered_excerpts.items() if k != -1
            }

            cluster_numbers = np.array(list(cluster_lengths.keys()))
            len_each_cluster = np.array(list(cluster_lengths.values()))

            kept_clusters = cluster_numbers[np.argsort(len_each_cluster)[::-1][:5]]

            # update dict_clustered_excerpts with the most important clusters
            dict_clustered_excerpts = {
                cluster: dict_clustered_excerpts[cluster] for cluster in kept_clusters
            }

        summarized_entries_per_cluster = []
        for one_cluster_entries in list(dict_clustered_excerpts.values()):
            if len(one_cluster_entries) > n_min_entries:
                summarized_entries_per_cluster.append(
                    get_summary_one_cluster(
                        one_cluster_entries,
                        severity_scores=None,
                        reliability_scores=None,
                        present_numbers_score=None,
                    )
                )

        summarized_entries = get_summary_one_cluster(summarized_entries_per_cluster)

    return summarized_entries


def order_dict(x):
    cleaned_x = {k: v for k, v in x.items() if str(k) != "[]" and str(v) != "{}"}

    if "General Overview" in list(cleaned_x.keys()):
        first_dict = {"General Overview": cleaned_x["General Overview"]}
        second_dict = {k: v for k, v in cleaned_x.items() if k != "General Overview"}
        y = {**first_dict, **second_dict}
        return y

    elif "['General Overview']" in list(cleaned_x.keys()):
        first_dict = {"[General Overview]": cleaned_x["['General Overview']"]}
        second_dict = {
            str(k): str(v) for k, v in cleaned_x.items() if k != "['General Overview']"
        }
        y = {**first_dict, **second_dict}
        return y

    else:
        return cleaned_x


def process_df_one_part(df: pd.DataFrame, col_name: str):
    """
    group df by col_name and get excerpts list for each different tag
    """
    df["tmp_tag_str"] = df[col_name].apply(str)

    grouped_df = df.groupby("tmp_tag_str", as_index=False)[
        [
            "excerpt",
            # "severity_scores",
            # "reliability_scores",
            # "present_numbers_score",
            "entry_id",
        ]
    ].agg(lambda x: list(x))

    grouped_df["len"] = grouped_df["excerpt"].apply(lambda x: len(x))
    grouped_df = grouped_df[grouped_df.len >= 5]
    grouped_df["n_clusters"] = grouped_df["len"].apply(get_number_of_clusters)

    """grouped_df.sort_values(by="len", ascending=False, inplace=True).drop(
        columns="tmp_tag_str", inplace=True
    )"""

    return grouped_df.sort_values(by="tmp_tag_str")


def get_summary_one_part(df, col_name: str, en_summary, n_min_entries: int = 10):
    """
    main function to get the summary for one part
    example:
        - col_name: sectors
        - we get a paragraph for each different sector using the function 'get_summary_one_row'
    """
    preprocessed_df = df[df[col_name].apply(lambda x: len(x) > 0)].copy()
    if len(preprocessed_df) < n_min_entries:
        return {}
    else:
        preprocessed_df = process_df_one_part(df, col_name)

        final_returns = {}
        for index, row in preprocessed_df.iterrows():
            tag_tmp = row.tmp_tag_str
            summarized_text = get_summary_one_row(row, en_summary)
            if len(summarized_text) > n_min_entries:
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
        lambda x: preprocesss_row(
            [
                item
                for item in x
                if item
                in [
                    "Humanitarian Conditions->Living Standards",
                    "Humanitarian Conditions->Physical And Mental Well Being",
                    "Humanitarian Conditions->Coping Mechanisms",
                ]
            ],
            1,
        )
    )

    return get_summary_one_part(one_affected_group_df, "non_num_hum_conds", en_summary)


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
        if save_summary:
            with open(f"first_section_report.json", "w") as fp:
                json.dump(summaries_impact_hum_conditions, fp)

        # generate report for part 1.2
        print("begin Shocks and impact of the crisis")
        summaries_impact_hum_conditions[
            "Shocks and impact of the crisis"
        ] = get_report_shocks_impacts(df_one_project, en_summary)
        if save_summary:
            with open(f"first_section_report.json", "w") as fp:
                json.dump(summaries_impact_hum_conditions, fp)

        # generate report for part 1.4
        print("begin Humanitarian conditions and severity of needs")
        summaries_impact_hum_conditions[
            "Humanitarian conditions and severity of needs"
        ] = get_report_hum_conditions(df_one_project, en_summary)
        if save_summary:
            with open(f"first_section_report.json", "w") as fp:
                json.dump(summaries_impact_hum_conditions, fp)

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
