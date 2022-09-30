import torch
from typing import List


def flatten(t: List[List]) -> List:
    """flatten list of lists"""
    return [item for sublist in t for item in sublist]


def process_tag(tags: List[str], tag_section: str):
    if tag_section == "sectors":
        return [f"sectors->{one_tag}" for one_tag in tags]
    else:  # subpillars
        return [f"{tag_section.replace('sub', '')}->{one_tag.split('->')[0]}" for one_tag in tags]

