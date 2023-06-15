female_linked_kwords = ["enceinte"]
neutral_to_be_removed = []

fr_to_be_removed_kwords = female_linked_kwords + neutral_to_be_removed

neutral2male = {
    "personne": "homme",
    "personnes": "hommes",
    "individu": "homme",
    "individus": "hommes",
    "enfant": "garçon",
    "enfants": "garçons",
}

neutral2female = {
    "personne": "femme",
    "personnes": "femmes",
    "individu": "femme",
    "individus": "femmes",
    "enfant": "fille",
    "enfants": "filles",
}

female2neutral = {
    "fille": "enfant",
    "filles": "enfants",
    "femme": "personne",
    "femmes": "personnes",
    "mère": "individu",
    "mères": "individus",
}

male2female = {
    "garçon": "fille",
    "garçons": "filles",
    "homme": "femme",
    "hommes": "femmes",
    "père": "mère",
    "pères": "mères",
}

male2neutral = {
    "garçon": "enfant",
    "garçons": "enfants",
    "homme": "personne",
    "hommes": "personnes",
    "père": "individu",
    "pères": "individus",
}

female2male = {v: k for k, v in male2female.items()}

male_mappings = {"female": male2female, "neutral": male2neutral}
female_mappings = {"male": female2male, "neutral": female2neutral}
neutral_mappings = {"male": neutral2male, "female": neutral2female}

male_all = {"keywords": list(male2female.keys()), "mappings": male_mappings}
female_all = {"keywords": list(male2female.values()), "mappings": female_mappings}
neutral_all = {"keywords": list(neutral2male.keys()), "mappings": neutral_mappings}

fr_biases_all = {"male": male_all, "female": female_all, "neutral": neutral_all}
