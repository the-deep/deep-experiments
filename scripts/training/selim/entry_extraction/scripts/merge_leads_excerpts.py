from typing import List, Dict
from datasets import load_dataset
import pandas as pd
from ast import literal_eval
from copy import copy
from utils import flatten

# MERGING EXCERPTS AND LEAD, USED BEFORE TRAININIG


def process_tag(tags: List[str], tag_section: str) -> List[str]:
    """
    Get list of sectors, pillars_1d and pillars_2d from each prediction.
    args:
        - 'tags': List of tags: example: tags=['Context->Politics'], tag_section='subpillars_1d'
        - 'tag_section': str: one of['sectors', 'subpillars_1d', 'subpillars_2d']

    Outputs:
        - Not the same processing for sectors and subpillars
        - Example of outputs= ['pillars_1d->Context->Politics']
    """
    if tag_section == "sectors":
        return [
            f"sectors->{one_tag}" for one_tag in tags if "NOT_MAPPED" not in one_tag
        ]
    else:  # subpillars
        return list(
            set(
                [
                    f"{tag_section.replace('sub', '')}->{one_tag.split('->')[0]}"
                    for one_tag in tags
                    if "NOT_MAPPED" not in one_tag
                ]
            )
        )


def get_excerpts_dict(excerpts_df) -> Dict[int, List[str]]:
    """
    Get for each entry_id the list of primary tags (sectors, pillars_1d and pillars_2d)
    """
    excerpts_df["primary_tags"] = excerpts_df.apply(
        lambda x: flatten(
            [
                process_tag(literal_eval(x[tag]), tag)
                for tag in ["sectors", "subpillars_1d", "subpillars_2d"]
            ]
        ),
        axis=1,
    )

    excerpts_df["primary_tags"] = excerpts_df["primary_tags"].apply(
        lambda x: x + ["is_relevant"] if len(x) > 0 else x
    )

    excerpts_dict = dict(zip(excerpts_df.entry_id, excerpts_df.primary_tags))

    return excerpts_dict


def add_tags_to_excerpt_sentence_indices(
    tagged_excerpts: List[Dict[str, int]], excerpts_dict: Dict[int, List[str]]
):
    """
    preprocessing function: add the tags to the excerpts
    """
    final_outputs = []
    for one_excerpt_vals in tagged_excerpts:
        new_val = copy(one_excerpt_vals)
        new_val.update({"tags": excerpts_dict[one_excerpt_vals["source"]]})
        final_outputs.append(new_val)

    return final_outputs


def get_training_dict(
    leads_data_path: str,
    excerpts_df_path: str,
    use_sample: bool,
    sample_percentage: float = 0.1,
):
    """
    Main function, used to ....
    """

    excerpts_df = pd.read_csv(excerpts_df_path)
    full_leads_data = (
        load_dataset("json", data_files=leads_data_path, split="train")
        .filter(
            lambda x: len(x["excerpts"]) > 0 and len(x["excerpt_sentence_indices"]) > 0
        )
        .to_dict()
    )

    n_leads = len(full_leads_data["id"])

    excerpts_dict = get_excerpts_dict(excerpts_df)
    full_tags_list = flatten(excerpts_dict.values())

    label_names = sorted(list(set(full_tags_list)))
    tagname_to_tagid = {tag_name: tag_id for tag_id, tag_name in enumerate(label_names)}

    kept_leads = []

    for i in range(n_leads):
        outputs_one_lead = {}

        # id
        outputs_one_lead["id"] = {
            "lead_id": full_leads_data["id"][i][0],
            "project_id": full_leads_data["id"][i][1],
        }

        # sentences
        outputs_one_lead["sentences"] = full_leads_data["sentences"][i]

        # excerpt_sentence_indices
        outputs_one_lead[
            "excerpt_sentence_indices"
        ] = add_tags_to_excerpt_sentence_indices(
            full_leads_data["excerpt_sentence_indices"][i], excerpts_dict
        )

        kept_leads.append(outputs_one_lead)

    if use_sample:
        n_samples = int(n_leads * sample_percentage)
        kept_leads = [kept_leads[i] for i in range(n_samples)]

    final_output = {
        "data": kept_leads,
        "tagname_to_tagid": tagname_to_tagid,
    }

    output_as_df = pd.DataFrame(
        [
            [
                final_output["data"],
                final_output["tagname_to_tagid"],
            ]
        ],
        columns=["data", "tagname_to_tagid"],
    )

    return output_as_df
