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

import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words())

sectors_list = [
    'Shelter',
    'Food Security',
    'Agriculture',
    'Education',
    'Health',
    'Logistics',
    'Protection',
    'Livelihoods',
    'WASH',
    'Nutrition']

affected_groups_list = ['Asylum Seekers', 'Host', 'IDP', 'Migrants', 'Refugees','Returnees']

#TODO: CLUSTERING WITH FIXED NUMBER OF CLUSTERS TO BE ABLE TO GENERATE LONGER TEXT

def flatten(t):
    return [item for sublist in t for item in sublist]

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
    return min((n_entries // 10) + 1, 3)
 
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
        #en_entries = translate_sentence(entries)
        MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
        model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
        tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

        inputs = tokenizer([entries], return_tensors="pt")

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

def translate_sentence(excerpt: str):
    translator = Translator()
    result = translator.translate(excerpt, dest='en')
    return result.text

def clean_excerpt(excerpt):

    # lower and remove punctuation
    new_excerpt = re.sub(r'[^\w\s]', '', excerpt).lower()
    # omit stop words
    new_excerpt = ' '.join([word for word in new_excerpt.split(' ') if word not in stop_words])
    return new_excerpt

def get_summary_one_row(row, en_summary):

    """
    function used for getting summary for one task
    """
    
    original_excerpts = row.excerpt
    severity_scores = row.severity_scores
    #top_n_sentences = row.n_clusters

    cleaned_excerpts = [clean_excerpt(one_excerpt) for one_excerpt in original_excerpts]
    cosine_similarity_matrix = get_similarity_matrix(cleaned_excerpts)

    graph_one_lang = build_graph(cosine_similarity_matrix)

    scores_pagerank = nx.pagerank(graph_one_lang)
    scores = {key: value * severity_scores[key] for key, value in scores_pagerank.items()}

    top_n_sentence_ids = np.argsort(np.array(list(scores.values())))[-10:]
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

    ranked_sentence = ' '.join([original_excerpts[id_tmp] for id_tmp in (top_n_sentence_ids)]) 
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
    
    preprocessed_df = process_df_one_part(df, col_name)

    final_returns = {}
    for index, row in preprocessed_df.iterrows():
        tag_tmp = row.tmp_tag_str
        summarized_text = get_summary_one_row(row, en_summary)
        print(summarized_text)
        final_returns[tag_tmp] = summarized_text

    return order_dict(final_returns)

def get_report_relation_other_sectors(df, en_summary):
    many_tags_df = df[
        (df.n_sectors>1) &
        (df.n_secondary_tags!=1)
        ].copy()
    many_tags_df['sectors'] = many_tags_df.sectors.apply(
        lambda x: preprocesss_row(x, 2)
    )
    #two_sectors_df = many_tags_df[many_tags_df.sectors.apply(lambda x: len(x)==2)]
    return get_summary_one_part(many_tags_df, 'sectors', en_summary)

def preprocesss_row(tags: List[str], n_tags: int):
    if len(tags)>n_tags:
        return ['General Overview']
    else:
        return tags

def get_report_secondary_tags(df, en_summary):

    df_secondary_tags = df[
        (df.n_sectors==1) & 
        (df.secondary_tags.apply(lambda x: len(x)>0)) 
    ].copy()

    df_secondary_tags['secondary_tags'] = df_secondary_tags['secondary_tags'].apply(lambda x: preprocesss_row(x, 1))

    return get_summary_one_part(df_secondary_tags, 'secondary_tags', en_summary)

def get_report_hum_needs(df, en_summary):
    hum_conds_df = df[
        (df.pillars_2d.apply(lambda x: 'Humanitarian Conditions' in x)) &
        (df.n_sectors==1) &
        (df.n_secondary_tags!=1)].copy()

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
    df['secondary_tags'] = df.apply(
        lambda x: flatten(
            [f'{col}->{x[col]}' for col in ['affected_groups_level_3', 'specific_needs_groups', 'age']]
        ), axis=1
    )
    df['n_secondary_tags'] = df.secondary_tags.apply(
        lambda x: len(x)
    )
    df['n_affected_groups'] = df.affected_groups.apply(
        lambda x: len(x)
    )

def get_report_one_sector(one_project_df: pd.DataFrame, tag: str, en_summary: bool):
    """
    section 3 each one
    """
    df_one_sector = one_project_df[one_project_df.sectors.apply(lambda x: tag in x)].copy()

    summaries_one_sector = {}
    
    # secondary tags
    summaries_one_sector['Most affected population groups'] = get_report_secondary_tags(df_one_sector, en_summary)
    
    # key trends
    summaries_one_sector['Key trends'] = get_report_hum_needs(df_one_sector, en_summary)

    #relation to other sectors
    summaries_one_sector['Needs, severity and linkages with other sectors'] = get_report_relation_other_sectors(df_one_sector, en_summary)

    return summaries_one_sector

def get_report_shocks_impacts(df_one_project, en_summary):
    """
    section 1.2
    """
    summary = {}
    impact_df = df_one_project[
        df_one_project.pillars_2d.apply(lambda x: 'Impact' in x)
    ].copy()

    # Impact on people
    impact_people_df = impact_df[
        impact_df.subpillars_2d.apply(lambda x: 'Impact->Impact On People' in x)
        ].copy()
    impact_people_df['sectors'] = impact_people_df['sectors'].apply(lambda x: preprocesss_row(x, 1))
    summary['Impact on people'] = get_summary_one_part(impact_people_df, 'sectors', en_summary)

    # Impact on systems and services
    impact_systems_services_df = impact_df[
        impact_df.subpillars_2d.apply(lambda x: 'Impact->Impact On Systems, Services And Networks' in x)
        ].copy()
    impact_systems_services_df['sectors'] = impact_systems_services_df['sectors'].apply(lambda x: preprocesss_row(x, 1))
    summary['Impact on systems and services'] = get_summary_one_part(impact_systems_services_df, 'sectors', en_summary)

    # Humanitarian Access, TODO: ADD NUMBERS HERE!!
    hum_access_df = df_one_project[
        df_one_project.pillars_1d.apply(lambda x: 'Humanitarian Access' in x)
        ].copy()
    hum_access_df['non_number_hum_access'] = hum_access_df['subpillars_1d'].apply(
        lambda x: preprocesss_row(
                [item for item in x if item in [
                    'Humanitarian Access->Physical Constraints',
                    'Humanitarian Access->Population To Relief',
                    'Humanitarian Access->Relief To Population'
                    ]
                ], 
            1)
    )
    summary['Humanitarian Access'] = get_summary_one_part(hum_access_df, 'non_number_hum_access', en_summary)

    return summary


def get_report_context_crisis(df_one_project: pd.DataFrame, en_summary: bool):
    """
    section 1.1
    """
    df_context = df_one_project[
        df_one_project.pillars_1d.apply(lambda x: 'Context' in x)
    ].copy()

    df_context['context_tags'] = df_context.subpillars_1d.apply(
        lambda x: preprocesss_row(
            [item.split('->')[1] for item in x if 'Context' in item],
            1)
    )

    return get_summary_one_part(df_context, 'context_tags', en_summary)

def get_hum_report_one_affected_group(df: pd.DataFrame, tag: str, en_summary: bool):
    """
    one pop in section 1.4, 
    TODO: IMPROVE IT!!
    """
    if tag=='Overall Tendencies':
        one_affected_group_df = df[df.n_affected_groups>1]
    else:
        one_affected_group_df = df[df.affected_groups.apply(lambda x: [tag] == x)]

    one_affected_group_df['non_num_hum_conds'] = one_affected_group_df.subpillars_2d.apply(
        lambda x: [
            item for item in x if item in [
                'Humanitarian Conditions->Living Standards',
                'Humanitarian Conditions->Physical And Mental Well Being',
                'Humanitarian Conditions->Coping Mechanisms']
        ]
    )

    return get_summary_one_part(one_affected_group_df, 'non_num_hum_conds', en_summary)
    

def get_report_hum_conditions(df_one_project: pd.DataFrame, en_summary: bool):
    """
    section 1.4
    """
    df_all_hum_conditions = df_one_project[df_one_project.pillars_2d.apply(lambda x: 'Humanitarian Conditions' in x)].copy()
    summary = {}
    
    # General Findings
    summary['Overall Tendencies'] = get_hum_report_one_affected_group(df_all_hum_conditions, 'Overall Tendencies', en_summary)
    
    # specific to each affected group
    for one_group in affected_groups_list:
        summary[one_group] = get_hum_report_one_affected_group(df_all_hum_conditions, one_group, en_summary)

    return summary


def get_report(full_df, project_id: int, en_summary: bool = True, save_summary: bool = True, use_sample: bool = False):
    """
    main function to get full report for sectoral analysis section
    """
    
    df_one_project = full_df[full_df.project_id==project_id].copy()    
    if use_sample: 
        df_one_project = df_one_project.sample(frac=0.2)
    if use_sample: 
        df_one_project['excerpt'] = df_one_project['excerpt'].apply(translate_sentence)
    
    preprocess_df(df_one_project)

    final_summary = {}


    # Part 1: Impact of the crisis and humanitarian conditions
    summaries_impact_hum_conditions = {}
    print('begin Context of the crisis')
    summaries_impact_hum_conditions['Context of the crisis'] = get_report_context_crisis(df_one_project, en_summary)

    print('begin Shocks and impact of the crisis')
    summaries_impact_hum_conditions['Shocks and impact of the crisis'] = get_report_shocks_impacts(df_one_project, en_summary)

    print('begin Humanitarian conditions and severity of needs')
    summaries_impact_hum_conditions[
        'Humanitarian conditions and severity of needs'
        ] = get_report_hum_conditions(df_one_project, en_summary)

    final_summary['Impact of the crisis and humanitarian conditions'] = summaries_impact_hum_conditions

    # Part 3: Sectoral analysis
    summaries_sectors = {}
    for one_sector in sectors_list:
        print(f'begin {one_sector}')
        
        summaries_sectors[one_sector] = get_report_one_sector(df_one_project, tag=one_sector, en_summary=en_summary)
    final_summary['Sectoral Analysis'] = summaries_sectors

    if save_summary:
        with open(f'report_sectors_{project_id}.json', 'w') as fp:
            json.dump(summaries_sectors, fp)

    return summaries_sectors