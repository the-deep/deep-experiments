import pandas as pd
import spacy

nlp = spacy.load("xx_ent_wiki_sm")
import numpy as np

import warnings

warnings.filterwarnings("ignore")

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapiExercises")

# SPECIFICATIONS:
# - one DataFrame = one lead
# TO RUN:
# from process_locations import get_locations
# data = get_locations(DATA_PATH)

# If string IS (EQUAL) prediction => omit prediction
not_equal_list = [
    "dc",
    "no",
    "south",
    "north",
    "west",
    "east",
    "Afrin",
    "cenral",
    "St",
    "unknown",
    "Sites",
    "",
    "north west",
    "north east",
    "Northwest",
    "northeast",
    "centre",
    'Bol',
    'lac',
    'Est',
    'ouest',
    'nord',
    'sud',
    "l’Est",
    "l’Ouest",
    "l'Est",
    "l'Ouest",
    'Grand Nord',
    'Sud-Ouest',
    'Nord-Ouest',
    "l'UE",
    "Sud - Ouest",
    "nord - Ouest",
    'Sud - est',
    'nord - est'
]
not_equal_list_up = [name.upper() for name in not_equal_list]

# If string IN prediction => omit all prediction
wrong_locations_list = [
    "école",
    "éducation",
    "insécurité",
    "adolescent",
    "Svp",
    "Government",
    "cluster",
    "shocking",
    "property",
    "housing",
    "feedback",
    "mechanism",
    "survey",
    "conduct",
    "song",
    "import",
    "fight",
    "song",
    "household",
    "individ",
    "broke",
    "chairm",
    "poor",
    "borderline",
    "accept",
    "flood",
    "nutrition",
    "water",
    "sanitat",
    "people",
    "hygiene",
    "detail",
    "report",
    "world",
    "complain",
    "unama",
    "covid",
    "gbv",
    'exploitation',
    'province',
    'provinces',
    'eau',
    'arrêté',
    'lac',
    'Gouvernement',
    'Réfug',
    "l'exploitation",
    'Camerounais',
    'Soudanais',
    'Nigérien',
    'Sur',
    'Femme',
    "l'ont",
    "besoin",
    'Centrafricains',
    'lake',
    'this',
    'canal',
    'Business',
    'Extrême',
    'analyse',
    'les',
    'état',
    'emploi',
    'Infantil',
    'pacífico',
    'afin',
    'approvision',
    'patrimoine',
    'covid',
    'haram',
    'entrée'
]
wrong_locations_list_up = [name.upper() for name in wrong_locations_list]



def word_is_valid(word: str, contains_invalid, equals_invalid) -> bool:
    """
    check whether a prediction is valid or not
    return boolean
    """
    if word == "":
        return False

    for item_invalid in contains_invalid:
        if item_invalid in word.upper():
            return False
    for item_invalid in equals_invalid:
        if item_invalid == word.upper():
            return False

    return True


def most_frequent(List):
    """
    Get most frequent element of list
    """
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


def get_place(
    text:str,
    contains_invalid=wrong_locations_list_up,
    equals_invalid=not_equal_list_up,
    augment_predictions=False,
):
    """
    Given a sentence, get the location
    """

    article = nlp(text)
    array_ents = np.array(article.ents, dtype=object)
    article_ents = np.array(["".join(str(x)) for x in array_ents], dtype=object)

    if len(article_ents) == 0:
        return "EMPTY ENTRY"

    labels = np.array([x.label_ for x in article.ents])
    bool_GPE = labels == "LOC"
    list_items = list(article_ents[bool_GPE].flatten())

    cleaned_list = []
    for item in list_items:
        if "[" in item:
            final_item = item[1:-1]
        else:
            final_item = item

        final_item = (
            final_item.replace("’s", "")
            .replace("'s", "")
            .replace("unknown", "")
            .replace("Somelis", "Somalia")
            .rstrip()
            .lstrip()
        )

        if word_is_valid(final_item, contains_invalid, equals_invalid):

            if augment_predictions:
                location = geolocator.geocode(final_item, language="en")
                if location is not None:
                    cleaned_list.append(location.address)
                else:
                    cleaned_list.append(final_item)
            else:
                cleaned_list.append(final_item)

    if len(cleaned_list) == 0:
        return "EMPTY ENTRY"

    else:
        return list(np.unique(cleaned_list))


def postprocess_predictions(predicted_tag:list, max_tag:str):
    
    postprocessed_tag = [list(np.unique(x+[max_tag])) if x != "EMPTY ENTRY" else [max_tag]\
                        for x in predicted_tag]

    return postprocessed_tag

def get_most_tagged_place(text:str):
    """
    Function to get the most tagged place: returns one entry
    """
    return most_frequent(get_place(text))

def get_locations(
    entries:list, max_place:str, augment_predictions=False
):
    """
    Main function: returns list of predictions for list of entries
    """
    predicted_tag = [get_place(x, augment_predictions=augment_predictions) for x in entries]
    
    postprocessed_tag = postprocess_predictions(predicted_tag, max_place)
    return postprocessed_tag
