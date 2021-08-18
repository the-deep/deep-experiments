import pandas as pd
import numpy as np
from itertools import chain
from nltk.corpus import wordnet

def str_in_list(List, name:str)->bool:
    return np.any([name.upper() in x.upper().split(' ') for x  in List])

languages = ['eng', 'fra', 'spa']

list_tags = ['Adult Female (18 to 59 years old)',
           'Adult Male (18 to 59 years old)',
           'Adult Unspecified gender (18-59 years old)',
           'Children/Youth Female (5 to 17 years old)',
           'Children/Youth Male (5 to 17 years old)',
           'Children/Youth Unspecified gender (5 to 17 years old)',
           'Infants/Toddlers (<5 years old)',
           'Older Persons Female (60+ years old)',
           'Older Persons Male (60+ years old)',
           'Older Persons Unspecified gender (60+ years old)']

def get_list(List):
    list_words = list(set(chain.from_iterable([word.lemma_names(lang)\
                                              for lang in languages\
                                              for syn in List
                                              for word in syn])))
    return [x.upper() for x in list_words]

def get_list_female():
    synonyms_gender = wordnet.synsets('gender')
    synonyms_enceinte = wordnet.synsets('mariage')

    synonyms_tot_female = [synonyms_gender, synonyms_enceinte]

    list_words_female = get_list(synonyms_tot_female)

    list_words_female.append('CONTRACEPT')
    list_words_female.append('PREGNAN')
    list_words_female.append('ENCEINT')
    list_words_female.append('SEX')
    list_words_female.append('VGB')
    list_words_female.append('FEMIN')
    
    return list_words_female

def get_list_male():
    
    return ['MASCUL']

def get_list_minor_with_gender():
    synonyms_boy = wordnet.synsets('boy')
    synonyms_girl = wordnet.synsets('girl')
    
    list_male = get_list([synonyms_boy])
    list_female = get_list([synonyms_girl])
    
    return list_male, list_female

def get_list_major_with_gender():
    synonyms_man = wordnet.synsets('man')
    synonyms_male = wordnet.synsets('male')
    synonyms_woman = wordnet.synsets('woman')
    synonyms_female = wordnet.synsets('female')
    
    list_male = get_list([synonyms_male, synonyms_man])
    list_male.append('MEN')
    
    list_female = get_list([synonyms_female, synonyms_woman])
    list_female.append('WOMEN')
    
    return list_male, list_female

def get_list_toddlers():
    #synonyms = wordnet.synsets('child')
    synonyms_malnutrition = wordnet.synsets('malnutrition')
    synonym_toddler = wordnet.synsets('toddler')
    synonym_infant = wordnet.synsets('infant')

    syn_tot_toddlers = [synonyms_malnutrition, synonym_toddler, synonym_infant]

    list_words = get_list(syn_tot_toddlers)
    list_words.append('INFANT')
    
    return list_words

def get_list_youth():
    synonyms = wordnet.synsets('child')
    synonyms_adolescent = wordnet.synsets('adolescent')
    synonyms_young = wordnet.synsets('young')
    synonyms_school = wordnet.synsets('school')

    list_no_gender = get_list([synonyms, synonyms_adolescent, synonyms_young, synonyms_school])
    
    return list_no_gender

def get_list_adult():

    synonyms = wordnet.synsets('adult')
    #synonyms_person = wordnet.synsets('person')
    #synonyms_population = wordnet.synsets('population')
    synonyms_man = wordnet.synsets('man')
    synonyms_male = wordnet.synsets('male')
    synonyms_woman = wordnet.synsets('woman')
    synonyms_female = wordnet.synsets('female')

    syn_tot_adults = [synonyms]

    list_words = get_list(syn_tot_adults)
    
    return list_words

def get_list_elder():
    synonyms = wordnet.synsets('elder')
    synonyms_old = wordnet.synsets('old')

    syn_tot_elders = [synonyms, synonyms_old]

    list_words = get_list(syn_tot_elders)
    
    return list_words

list_women_all = get_list_female()
list_men_all = get_list_male()

list_men_minor, list_women_minor = get_list_minor_with_gender()
list_men_major, list_women_major = get_list_major_with_gender()

list_toddlers = get_list_toddlers()
list_youth = get_list_youth()
list_adult = get_list_adult()
list_elder = get_list_elder()

def list_in_sent(sent, list_words)->bool:
    return np.any([word.upper() in sent for word in list_words])

def return_demographic_groups_one_sent(text):
    sent = text.upper()
    final_tags = []
    
    if list_in_sent(sent, list_toddlers):
        final_tags.append('Infants/Toddlers (<5 years old)')
        
    if list_in_sent(sent, list_youth):
        final_tags.append('Children/Youth Male (5 to 17 years old)')
        final_tags.append('Children/Youth Female (5 to 17 years old)')
        
    if list_in_sent(sent, list_adult):
        final_tags.append('Adult Female (18 to 59 years old)')
        final_tags.append('Adult Male (18 to 59 years old)')
        
    if list_in_sent(sent, list_elder):
        final_tags.append('Older Persons Female (60+ years old)')
        final_tags.append('Older Persons Male (60+ years old)')
        
    if list_in_sent(sent, list_women_all):
        final_tags.append('Children/Youth Female (5 to 17 years old)')
        final_tags.append('Adult Female (18 to 59 years old)')
        final_tags.append('Older Persons Female (60+ years old)')
        
    if list_in_sent(sent, list_men_all):
        final_tags.append('Children/Youth Male (5 to 17 years old)')
        final_tags.append('Adult Male (18 to 59 years old)')
        final_tags.append('Older Persons Male (60+ years old)')

    if list_in_sent(sent, list_men_minor):
        final_tags.append('Children/Youth Male (5 to 17 years old)')
        #final_tags.append('Infants/Toddlers (<5 years old)')
        
    if list_in_sent(sent, list_women_minor):
        final_tags.append('Children/Youth Female (5 to 17 years old)')
        #final_tags.append('Infants/Toddlers (<5 years old)')
        
    if list_in_sent(sent, list_men_major):
        final_tags.append('Older Persons Male (60+ years old)')
        final_tags.append('Adult Male (18 to 59 years old)')
        
    if list_in_sent(sent, list_women_major):
        final_tags.append('Older Persons Female (60+ years old)')
        final_tags.append('Adult Female (18 to 59 years old)')
        
    if len(final_tags)==0:
        final_tags.append('UNKNOWN')
        
    return np.unique(final_tags)

def get_demographic_groups (import_data, data_path:str=None, data:pd.DataFrame=None):
    if import_data:
        try:
            demo_groups_data = pd.read_csv(data_path, index_col=0)[['excerpt']]
        except:
            demo_groups_data = pd.read_csv(data_path, index_col=0, lineterminator='\n')[['excerpt']]
    else:
        demo_groups_data = data[['excerpt', 'tag_value']].dropna()

    demo_groups_data['predicted_tag'] =\
         demo_groups_data['excerpt'].apply(lambda x: return_demographic_groups_one_sent (x))

    return demo_groups_data
