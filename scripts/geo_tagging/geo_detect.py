import re
import spacy
import numpy as np
import pandas as pd
from unidecode import unidecode
from geopy.geocoders import Nominatim

# init
geo_areas = pd.read_csv("geo_areas_list.csv", usecols=[
                        "title"])["title"].dropna()
geo_areas_pp = set()
for loc in geo_areas:
    loc = unidecode(loc).lower()
    if loc.endswith(" (cloned)"):
        geo_areas_pp.update([loc[:-len(" (cloned)")]])
    else:
        geo_areas_pp.update([loc])
geo_areas = geo_areas_pp
nlp = spacy.load("xx_ent_wiki_sm")
geolocator = Nominatim(user_agent="geoapiExercises")


def preprocess(text):
    text = re.sub("([\[\].,!?():])", r" \1 ", text)
    text = re.sub("\s+", " ", text)
    text = text.replace("’", "'").replace("Somelis", "Somalia").replace(
        "“", '"').replace("”", '"').strip()
    text = text.lower()
    return text


def detect_locs(text, static_list_filter=False, osm_filter=True, aug_preds=True, prep=True, lang="en"):
    if prep:
        text = preprocess(text)
    text_processed = nlp(text)
    ents = np.array(["".join(str(x)) for x in text_processed.ents],
                    dtype=str)
    locs = np.array([x.label_ for x in text_processed.ents],
                    dtype=str) == "LOC"
    locs = list(ents[locs].flatten())
    ##
    locs_processed = []
    for loc in locs:
        loc = loc[1:-1] if loc.startswith("[") else loc
        loc = loc.replace("unknown", "")

        if aug_preds or osm_filter:
            loc_aug = geolocator.geocode(loc, language=lang)
            if aug_preds and osm_filter and loc_aug is not None:
                locs_processed.extend(
                    [l for l in loc_aug.address.split(
                        ", ") if not l.isnumeric()]
                )
            elif (
                not aug_preds
                and osm_filter
                and loc_aug is not None
                or not osm_filter
            ):
                locs_processed.append(loc)
        else:
            locs_processed.append(loc)
    if static_list_filter:
        locs_processed = [loc for loc in locs_processed if unidecode(
            loc).lower() in geo_areas]
    locs_processed = list(set(locs_processed))
    return locs_processed
