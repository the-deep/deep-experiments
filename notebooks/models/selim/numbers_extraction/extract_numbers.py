from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

kept_tags = {
    'people': '',
    'severly': '',
    'women': 'gender->Female',
    'men': 'gender->Male',
    'infants': 'age->Infants/Toddlers (<5 years old)',
    'children': 'age->Children/Youth (5 to 17 years old)',
    'adults': 'age->Adult (18 to 59 years old)',
    'elderly': 'age->Older Persons (60+ years old)',
    'handicapped': 'specific_needs_groups->Persons with Disability',
    'displaced': 'affected_groups->IDP',
    'returnees': 'affected_groups->Returnees',
    'refugees': 'affected_groups->Refugees',
    'migrants': 'affected_groups->Migrants',
}

sec_tags_mapper = {
    'Female': 'gender->Female',
    'Male': 'gender->Male',
    'Infants/Toddlers (<5 years old)': 'age->Infants/Toddlers (<5 years old)',
    'Children/Youth (5 to 17 years old)': 'age->Children/Youth (5 to 17 years old)',
    'Adult (18 to 59 years old)': 'age->Adult (18 to 59 years old)',
    'Older Persons (60+ years old)': 'age->Older Persons (60+ years old)',
    'Persons with Disability': 'specific_needs_groups->Persons with Disability',
    'IDP': 'affected_groups->IDP',
    'Returnees': 'affected_groups->Returnees',
    'Refugees': 'affected_groups->Refugees',
    'Migrants': 'affected_groups->Migrants'
}

treated_cases = list(sec_tags_mapper.keys())

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def get_numbers(df: pd.DataFrame, min_ratio: float = 0.7):

    model_name = 'deepset/roberta-large-squad2'

    returns_dict = {}
    df_copy = df.copy()
    df_copy['secondary_tags'] = df_copy.apply(
        lambda x: [sec_tags_mapper[item] for item in x if item in treated_cases], axis=1
    )
    
    for one_pop, related_tag in kept_tags.items():
        one_pop_df = df_copy.copy()
        """if related_tag!='':
            one_pop_df = one_pop_df[
                one_pop_df.secondary_tags.apply(lambda x: related_tag in x)
            ]"""
        print(f'begin for {one_pop}')

        # define Question Answering Pipeline
        qa_model = pipeline(model=model_name, tokenizer=model_name, task="question-answering")

        # define specific question depending on population
        question_one_project = f'How many {one_pop} need humanitarian assistance in Ukraine?'
        answers = []
        scores = []

        # generate answers and confidentiality of answer for each excerpt
        for one_excerpt in tqdm(one_pop_df.excerpt):
        
            raw_response = qa_model(question = question_one_project, context = one_excerpt)

            answers.append(raw_response['answer'])
            scores.append(raw_response['score'])

        one_pop_df['answer'] = answers
        one_pop_df['confidence'] = scores

        # keep only relevant answers
        one_pop_df = one_pop_df.sort_values(by='confidence', ascending=False, inplace=False)
        one_pop_df = one_pop_df[
            (one_pop_df.confidence>min_ratio) & 
            (one_pop_df.answer.apply(has_numbers))
            ]

        if len(one_pop_df)>0:
                
            # keep the most confident answer
            returned_answer = one_pop_df.groupby(
                'answer', as_index=False
                )['confidence'].apply(sum).sort_values(by='confidence', ascending=False).iloc[0].answer

            returns_dict[one_pop] = returned_answer

        else:
            returns_dict[one_pop] = 'UNKNOWN'

        with open(f'numbers_returns.json', 'w') as fp:
            json.dump(returns_dict, fp)

    return returns_dict