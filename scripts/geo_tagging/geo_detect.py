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


def detect_locs(text, ret_locs_in_db=True, aug_preds=False):
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
        loc = (loc.replace("’s", "").replace("'s", "").replace(
            "unknown", "").replace("Somelis", "Somalia").strip())

        if aug_preds:
            loc_aug = geolocator.geocode(loc, language="en")
            if loc_aug is not None:
                locs_processed.append(loc_aug.address)
            else:
                locs_processed.append(loc)
        else:
            locs_processed.append(loc)
    if ret_locs_in_db:
        return [loc for loc in locs_processed if unidecode(loc).lower() in geo_areas]
    return locs_processed
