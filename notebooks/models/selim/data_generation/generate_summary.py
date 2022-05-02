import networkx as nx
from ast import literal_eval
from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer
from googletrans import Translator
import json

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
    return min((n_entries // 10) + 1, 4)
 
def process_df_one_part(df: pd.DataFrame, col_name: str):
    
    df_copy = df[df[col_name].apply(lambda x: len(x)>0)].copy()
    df_copy['tmp_tag_str'] = df_copy[col_name].apply(str)

    grouped_df = df_copy.groupby('tmp_tag_str', as_index=False)[['entry_id', 'excerpt', 'severity_scores']].agg(lambda x: list(x))

    grouped_df['len'] = grouped_df['entry_id'].apply(lambda x: len(x))
    grouped_df = grouped_df[grouped_df.len>=5]
    grouped_df['n_clusters'] = grouped_df['len'].apply(get_number_of_clusters)

    grouped_df.sort_values(by='len', ascending=False, inplace=False).drop(columns='tmp_tag_str', inplace=False)

    return grouped_df


def get_severity_score(item: List[str]):
    severity_mapper = {
        'Critical': 1,
        'Major': 0.75,
        'Of Concern': 0.5
    }

    if len(item) == 0:
        return 0.5
    severity_name = item[0]
    if severity_name in list(severity_mapper.keys()):
        return severity_mapper[severity_name]
    else:
        return 0.5

def get_similarity_matrix(texts):

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings_all_sentences = model.encode(texts)

    similarity = cosine_similarity(embeddings_all_sentences, embeddings_all_sentences)
    return similarity


def t2t_generation(entries: str, english_returns: bool):

    if english_returns:
        en_entries = translate_sentence(entries)
        MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
        model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
        tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

        inputs = tokenizer([en_entries], return_tensors="pt")

        # Generate Summary
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, num_beam_groups=2)
        summarized_entries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


    else:
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        model = T5ForConditionalGeneration.from_pretrained("t5-large")

        changed_text  = f'summarize: {entries}'
        input_ids = tokenizer(changed_text, return_tensors="pt", truncation=False).input_ids
        outputs = model.generate(input_ids)
        summarized_entries = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summarized_entries

def translate_sentence(excerpt: str, destination_language: str = 'en'):
    translator = Translator()
    result = translator.translate(excerpt, dest=destination_language)
    return result.text

def get_summary_one_row(row, en_summary):

    """
    function used for getting summary for one task
    """
    
    original_excerpt = row.excerpt
    severity_scores = row.severity_scores
    #top_n_sentences = row.n_clusters

    cosine_similarity_matrix = get_similarity_matrix(original_excerpt)

    graph_one_lang = build_graph(cosine_similarity_matrix)

    scores_pagerank = nx.pagerank(graph_one_lang)
    scores = {key: value * severity_scores[key] for key, value in scores_pagerank.items()}

    top_n_sentence_ids = np.argsort(np.array(list(scores.values())))[::-1][:25]
    """used_ids = []
    final_summary = []
    for id_tmp in top_n_sentence_ids:
        row_id = cosine_similarity_matrix[id_tmp, :]
        top_id_row = np.argsort(row_id)[::-1]
        top_id_row = [id for id in top_id_row if id not in used_ids and id not in top_n_sentence_ids][:2]

        top_2_id_row = [id_tmp] + top_id_row

        used_ids += top_2_id_row

        ranked_sentence = ' '.join([original_excerpt[id_tmp] for id_tmp in (top_2_id_row)]) 
            
        summarized_entries = t2t_generation(ranked_sentence, en_summary)
        capitalized_summaries = ' '.join([sent.capitalize() for sent in nltk.tokenize.sent_tokenize(summarized_entries)])
        final_summary.append(capitalized_summaries)"""

    ranked_sentence = ' '.join([original_excerpt[id_tmp] for id_tmp in (top_n_sentence_ids)]) 
    summarized_entries = t2t_generation(ranked_sentence, en_summary)
    """sentences = nltk.tokenize.sent_tokenize(summarized_entries)
    final_summary_str = ' '.join([sent.capitalize() for sent in sentences])"""
    
    #final_summary_str = ' '.join(final_summary)
    
    return summarized_entries

def order_dict(x):
    if 'General Overview' in list(x.keys()):
        first_dict = {'General Overview': x['General Overview']}       
        second_dict = {k:v for k,v in x.items() if k!='b'}
        y = {**first_dict, **second_dict}
        return y
    else:
        return x

def get_summary_one_part(df, col_name: str, en_summary):

    print(f'BEGIN {col_name}')
    
    preprocessed_df = process_df_one_part(df, col_name)

    final_returns = {}
    for index, row in preprocessed_df.iterrows():
        tag_tmp = row.tmp_tag_str
        print(f'For: {tag_tmp}:')
        summarized_text = get_summary_one_row(row, en_summary)
        print(summarized_text)
        final_returns[tag_tmp] = summarized_text

    return order_dict(final_returns)

def get_report_relation_other_sectors(df, en_summary):
    many_tags_df = df[df.n_sectors>1].copy()
    many_tags_df['sectors'] = many_tags_df.sectors.apply(
        lambda x: preprocesss_row(
            [item for item in x if item!='Cross'], 
            2
            )
    )
    #two_sectors_df = many_tags_df[many_tags_df.sectors.apply(lambda x: len(x)==2)]
    return get_summary_one_part(many_tags_df, 'sectors', en_summary)

def preprocesss_row(tags: List[str], n_tags: int):
    if len(tags)>n_tags:
        return ['General Overview']
    else:
        return tags

def get_report_relation_affected_pops(df, en_summary):

    final_dict = {}
    for col_tmp in ['age', 'specific_needs_groups', 'affected_groups']:
        df[col_tmp] = df[col_tmp].apply(lambda x: preprocesss_row(x, 1))
        final_dict.update(get_summary_one_part(df.copy(), col_tmp, en_summary))

    return final_dict

def get_report_hum_needs(df, en_summary):
    hum_conds_df = df[df.pillars_2d.apply(lambda x: 'Humanitarian Conditions' in x)].copy()
    hum_conds_df['pillars_1d_no_casualties'] = hum_conds_df.pillars_1d.apply(
        lambda x: preprocesss_row(
            list(set([item for item in x if item!='Casualties'])),
            1
        )
    )

    return get_summary_one_part(hum_conds_df, 'pillars_1d_no_casualties', en_summary)

def preprocess_df(df):
    """
    inplace!!
    """
    for col in ['sectors', 'affected_groups_level_3', 'specific_needs_groups', 'age', 'gender', 'subpillars_1d', 'subpillars_2d', 'severity', 'geo_location']:
        df[col] = df[col].apply(
            lambda x: list(set([item for item in custom_eval(x)]))
            )
    df['pillars_2d'] = df.subpillars_2d.apply(lambda x: [item.split('->')[0] for item in x])
    df['pillars_1d'] = df.subpillars_1d.apply(lambda x: [item.split('->')[0] for item in x])
    df['affected_groups'] = df.affected_groups_level_3.apply(
        lambda x: [item for item in x if item not in ['None', 'Others of Concern']]
    )
    df['sectors'] = df.sectors.apply(
        lambda x: [item for item in x if item not in ['Cross']]
    )
    df['n_sectors'] = df.sectors.apply(
        lambda x: len(x)
    )
    df['severity_scores'] = df['severity'].apply(lambda x: get_severity_score(x))

def get_report(full_df, tag: str, project_id: int, en_summary: bool = True, save_summary: bool = True):
    """
    main function to get full report
    """
    
    df_one_sector = full_df[(full_df.project_id==project_id) & (full_df.sectors.apply(lambda x: tag in x))].copy()    
    preprocess_df(df_one_sector)

    summaries = {}
    
    # secondary tags
    summaries['affected_specific'] = get_report_relation_affected_pops(df_one_sector, en_summary)
    
    # key trends
    summaries['key_trends'] = get_report_hum_needs(df_one_sector, en_summary)

    #relation to other sectors
    summaries['other_sectors'] = get_report_relation_other_sectors(df_one_sector, en_summary)

    if save_summary:
        with open(f'report_{tag}_{project_id}.json', 'w') as fp:
            json.dump(summaries, fp)

    return summaries
