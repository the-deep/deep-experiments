female_linked_kwords = [
    "pregnan",
    "menstr",
    "lactat",
    "her",
]  # 'her' can be seen as her object, or I told her, which have different translations to man (his, him)
neutral_to_be_removed = [
    "them",
    "in-person",
    "those",
]  # can refer to objects, not people
# general_to_be_removed = ["king", "queen", "prince"]
gramatically_incorrect = [
    "she is",
    "he is",
    "they are",
    "she was",
    "he was",
    "they were",
]


en_to_be_removed_kwords = (
    female_linked_kwords + neutral_to_be_removed + gramatically_incorrect
)

neutral_kwords = [
    "person",
    "persons",
    "people",
    "child",
    "children",
    "individual",
    "individuals",
    "their",
]

male_kwords = [
    "he",
    "men",
    "man",
    "male",
    "males",
    "boy",
    "boys",
    "husband",
    "husbands",
    "father",
    "fathers",
    "him",
    "his",
]
female_kwords = [
    "she",
    "women",
    "woman",
    "female",
    "females",
    "girl",
    "girls",
    "wife",
    "wives",
    "mother",
    "mothers",
]

neutral2male = {
    "person": "man",
    "persons": "men",
    "people": "men",
    "child": "boy",
    "children": "boys",
    "individual": "man",
    "individuals": "men",
    "their": "his",
    "they": "he",
}

neutral2female = {
    "person": "woman",
    "people": "women",
    "child": "girl",
    "children": "girls",
    "persons": "women",
    "individual": "woman",
    "individuals": "women",
    "their": "her",
    "they": "she",
}

male2female = {
    "boy": "girl",
    "boys": "girls",
    "he": "she",
    "male": "female",
    "males": "females",
    "husband": "wife",
    "husbands": "wives",
    "men": "women",
    "man": "woman",
    "father": "mother",
    "fathers": "mothers",
    "him": "her",
    "his": "her",
}
female2male = {v: k for k, v in male2female.items() if v != "her"}

male2neutral = {
    "boy": "child",
    "boys": "children",
    "he": "they",
    "male": "person",
    "males": "people",
    "husband": "person",
    "husbands": "people",
    "men": "people",
    "man": "person",
    "father": "person",
    "fathers": "people",
    "him": "them",
    "his": "their",
}
female2neutral = {
    kw_female: male2neutral[kw_male] for kw_female, kw_male in female2male.items()
}

male_mappings = {"female": male2female, "neutral": male2neutral}
female_mappings = {"male": female2male, "neutral": female2neutral}
neutral_mappings = {"male": neutral2male, "female": neutral2female}

male_all = {"keywords": male_kwords, "mappings": male_mappings}
female_all = {"keywords": female_kwords, "mappings": female_mappings}
neutral_all = {"keywords": neutral_kwords, "mappings": neutral_mappings}

en_biases_all = {"male": male_all, "female": female_all, "neutral": neutral_all}
