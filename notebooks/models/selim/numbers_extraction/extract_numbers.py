from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm


questions = {
   'Displacement->Type/Numbers/Movements': 'How many people have been displaced?', 
   'Capacities & Response->Number Of People Reached/Response Gaps': 'How many people have been reached?',
   'Impact->Number Of People Affected': 'How many people are affected?',
   'Humanitarian Conditions->Number Of People In Need': 'How many people are in need of humanitarian assistance?',
   'Humanitarian Access->Number Of People Facing Humanitarian Access Constraints/Humanitarian Access Gaps': 'How many people are facing humanitarian acceess gaps or constrainsts?',
   'At Risk->Number Of People At Risk': 'How many people are at risk?'
}

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

def get_response(df: pd.DataFrame, n_highest_answers: int = 5):

    model_name = 'deepset/roberta-large-squad2'

    returns_dict = {}
    df_copy = df.copy()
    df_copy['secondary_tags'] = df_copy.apply(
        lambda x: [sec_tags_mapper[item] for item in x if item in treated_cases], axis=1
    )
    
    for one_pop, related_tag in kept_tags.items():
        one_pop_df = df_copy.copy()
        if related_tag!='':
            one_pop_df = one_pop_df[
                one_pop_df.secondary_tags.apply(lambda x: related_tag in x)
            ]
        print(f'begin for {one_pop}')

        qa_model = pipeline(model=model_name, tokenizer=model_name, task="question-answering")

        question_one_project = f'How many {one_pop} need humanitarian assistance?'
        answers = []
        scores = []
        for one_excerpt in tqdm(one_pop_df.excerpt):
        
            raw_response = qa_model(question = question_one_project, context = one_excerpt)

            answers.append(raw_response['answer'])
            scores.append(raw_response['score'])

        one_pop_df[f'answer'] = answers
        one_pop_df[f'confidence'] = scores

        #confidence_names = [name for name in one_pop_df.columns if 'confidence' in name]

        # keep only relevant ones
        one_pop_df = one_pop_df.sort_values(by='confidence', ascending=False, inplace=False).head(n_highest_answers)
        #one_pop_df = one_pop_df[one_pop_df.confidence>0.7]
        one_pop_df['confidence'] = one_pop_df['confidence'].apply(lambda x: np.round(x, 2))

        returns_dict[one_pop] = one_pop_df

    return returns_dict