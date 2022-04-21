import networkx as nx
from ast import literal_eval
from typing import List
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

def custom_eval(x):
    if str(x)=='nan':
        return []
    if str(x)=='[None]':
        return []
    if type(x)==list:
        return x
    else:
        return literal_eval(x)


def build_graph(cosine_similarity_matrix):
    """
    function to build graoh from similarity matrix
    """
    graph_one_lang = nx.Graph()
    matrix_shape = cosine_similarity_matrix.shape
    for i in range (matrix_shape[0]):
        for j in range (matrix_shape[1]):
            #do only once
            if i < j:
                sim = cosine_similarity_matrix[i, j]
                graph_one_lang.add_edge(i, j, weight=sim)
                graph_one_lang.add_edge(j, i, weight=sim)

    return graph_one_lang

def get_number_of_clusters(n_entries: int):
    if n_entries < 5:
        return 1
    else:
        return n_entries // 5

def process_df(df: pd.DataFrame, col_name: str):

    df_copy = df.copy()
    df_copy['tmp_tag_str'] = df_copy[col_name].apply(str)

    grouped_df = df_copy.groupby('tmp_tag_str', as_index=False)[['entry_id', 'excerpt', 'severity_score']].agg(lambda x: list(x))

    grouped_df['len'] = grouped_df['entry_id'].apply(lambda x: len(x))
    grouped_df = grouped_df[grouped_df.len>=3]
    grouped_df['severity_score'] = grouped_df.severity.apply(get_severity_score)

    """grouped_df['sorted_severity_scores'] = grouped_df.severity_score.apply(
        lambda x: np.argsort(x)[::-1]
    )

    for col_tmp in ['entry_id', 'excerpt', 'severity_score']:
        grouped_df[col_tmp] = grouped_df.apply(
            lambda x: [x[col_tmp][x['sorted_severity_scores'][i]] for i in range (x.len)], axis=1
        )"""

    grouped_df.sort_values(by='len', ascending=False, inplace=False).drop(columns='tmp_tag_str', inplace=False)

    return grouped_df#[['excerpt', 'n_clusters']]


def get_severity_score(item: List[str]):
    severity_mapper = {
        'Critical': 1,
        'Major': 0.75,
        'Of Concern': 0.5
    }
    if len(item) == 0:
        return 0.5
    severity_name = item[0]
    if severity_name in severity_mapper.keys():
        return severity_mapper[severity_name]
    else:
        return 0.5

def get_similarity_matrix(texts):

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings_all_sentences = model.encode(texts)

    similarity = cosine_similarity(embeddings_all_sentences, embeddings_all_sentences)
    return similarity


def t2t_generation(entries):
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    model = T5ForConditionalGeneration.from_pretrained("t5-large")

    changed_text  = f'summarize: {entries}'
    input_ids = tokenizer(changed_text, return_tensors="pt", truncation=False).input_ids
    outputs = model.generate(input_ids, max_length=256)
    summarized_entries = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summarized_entries

"""def get_sentences_to_omit(original_tweets: List[str]):
    
    cosine_similarity_matrix = get_similarity_matrix(original_tweets)
    
    #cosine_similarity_matrix
    too_similar_ids = np.argwhere(cosine_similarity_matrix > 0.993)

    sentences_to_omit = []
    for pair_ids in too_similar_ids:
        if pair_ids[0]<pair_ids[1]:
            sentences_to_omit.append(pair_ids[1])

    return sentences_to_omit"""