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
    "St",
    "unknown",
    "Sites",
    "",
    "north west",
    "north east",
    "Northwest",
    "northeast",
    "centre",
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
    text,
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


def postprocess_predictions(one_lead_data: pd.DataFrame):

    bool_empty_leads = one_lead_data.predicted_tag == "EMPTY ENTRY"

    # Create two separate tmp DataFrames: used to get most tagged entry
    empty_tags_df_lead = one_lead_data[bool_empty_leads]
    filled_tags_df_lead = one_lead_data[~bool_empty_leads]

    total_list = list(filled_tags_df_lead.predicted_tag)
    total_list = [val for sublist in total_list for val in sublist]

    if len(total_list) >= 1:
        max_tag = most_frequent(total_list)
    else:
        max_tag = "UNKNOWN"

    # Add most tagged entries to all the predictions
    empty_tags_df_lead.loc[:, "predicted_tag"] = empty_tags_df_lead.loc[:, "predicted_tag"].apply(
        lambda x: [max_tag]
    )

    filled_tags_df_lead["predicted_tag"].apply(lambda x: x.append(max_tag))
    filled_tags_df_lead["predicted_tag"] = filled_tags_df_lead["predicted_tag"].apply(
        lambda x: np.unique(x)
    )

    return pd.concat([empty_tags_df_lead, filled_tags_df_lead])


def get_locations(
    import_data, data_path: str = None, data: pd.DataFrame = None, augment_predictions=False
):
    if import_data:
        geo_location_data = pd.read_csv(data_path, index_col=0)[["excerpt"]].dropna()
    else:
        geo_location_data = data[["excerpt", "tag_value"]].dropna()

    geo_location_data["predicted_tag"] = geo_location_data["excerpt"].apply(
        lambda x: get_place(x, augment_predictions=augment_predictions)
    )

    final_df = postprocess_predictions(geo_location_data)
    return final_df
