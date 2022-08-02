import re
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict
import math
import operator



def order_dict(x):
    cleaned_x = {k: v for k, v in x.items() if str(k) != "[]" and str(v) != "{}"}

    if "General Overview" in list(cleaned_x.keys()):
        first_dict = {"General Overview": cleaned_x["General Overview"]}
        second_dict = {k: v for k, v in cleaned_x.items() if k != "General Overview"}
        y = {**first_dict, **second_dict}
        return y

    elif "['General Overview']" in list(cleaned_x.keys()):
        first_dict = {"[General Overview]": cleaned_x["['General Overview']"]}
        second_dict = {
            str(k): v for k, v in cleaned_x.items() if k != "['General Overview']"
        }
        y = {**first_dict, **second_dict}
        return y

    else:
        return cleaned_x


def omit_punctuation(text):
    # layout clean
    clean_key = text.replace("'", "").replace("[", "").replace("]", "")

    # omit pillar of any
    if "->" in clean_key:
        clean_key = clean_key.split("->")[1]

    return clean_key


def clean_characters(text):
    # clean for latex characters
    latex_text = text.replace("%", "\%").replace("$", "\$")

    # strip punctuation
    latex_text = re.sub(r'\s([?.!"](?:\s|$))', r"\1", latex_text)

    return latex_text


"""def update_df(new_data):
    if type(new_data) is str:
        try:
            new_data = literal_eval(new_data)
        except Exception:
            new_data = new_data

    returned_df = pd.DataFrame()
    for one_paragraph in new_data:
        raw_entries = one_paragraph
        one_paragraph_df = pd.DataFrame(
            list(zip(raw_entries["entry_id"], raw_entries["excerpt"])),
            columns=["entry_id", "excerpt"],
        )
        returned_df = returned_df.append(one_paragraph_df)
    return returned_df"""


def print_df(ranked_sentence):
    return ("\n  \hrule \n").join(ranked_sentence)


def update_outputs_list(final_report, final_raw_outputs, new_data):
    if type(new_data) is str:
        final_report += new_data
        # final_raw_outputs += new_data
    else:

        for one_paragraph in new_data:
            # raw_entries = one_paragraph["raw_outputs"]
            # final_raw_outputs += print_df(raw_entries["excerpt"])
            final_report += one_paragraph

    return final_report, final_raw_outputs


def get_dict_items(items, final_report, final_raw_outputs):

    returned_df = pd.DataFrame()

    first_key, first_value = items
    dict_treated = order_dict(first_value)

    if len(dict_treated) > 1:

        for key, value in dict_treated.items():

            final_report, final_raw_outputs = update_outputs_list(
                final_report,
                final_raw_outputs,
                str("\paragraph{" + omit_punctuation(key) + "}\n"),
            )

            final_report, final_raw_outputs = update_outputs_list(
                final_report, final_raw_outputs, value
            )

            final_report, final_raw_outputs = update_outputs_list(
                final_report, final_raw_outputs, "\n \n"
            )

            # df_one_paragraph = update_df(value)
            # df_one_paragraph["paragraph"] = f"{first_key}->{key}"
            # returned_df = returned_df.append(df_one_paragraph)

    elif len(dict_treated) == 1:
        value = list(dict_treated.values())[0]
        key = list(dict_treated.keys())[0]

        final_report, final_raw_outputs = update_outputs_list(
            final_report, final_raw_outputs, value
        )
        final_report, final_raw_outputs = update_outputs_list(
            final_report, final_raw_outputs, "\n \n"
        )

        # df_one_paragraph = update_df(value)
        # df_one_paragraph["paragraph"] = f"{first_key}->{key}"
        # returned_df = returned_df.append(df_one_paragraph)

    return final_report, final_raw_outputs, returned_df


############################################ VIZU FUNCTIONS #####################################################


def flatten(t):
    """
    flatten a list of lists.
    """
    return [item for sublist in t for item in sublist]


def custom_eval(x):
    if str(x) == "nan":
        return []
    if str(x) == "[None]":
        return []
    if type(x) == list:
        return x
    else:
        return literal_eval(x)


def get_vizu_1_lvl(col: pd.Series, col_name: str):

    whole_tag = sorted(flatten(col.apply(custom_eval)))
    tot_n_entries = len(col)
    n_labels = len(set(whole_tag))

    if n_labels > 10:
        width = 6
    elif n_labels > 3:
        width = 4
    else:
        width = 2

    fig, ax = plt.subplots(figsize=(width * 2, width))

    pillar_counts = dict(Counter(whole_tag))
    proportions = {
        tag_name: round(100 * tag_count / tot_n_entries, 1)
        for tag_name, tag_count in pillar_counts.items()
    }

    y = list(proportions.keys())
    x = list(proportions.values())

    # axes[j].set_title(f'{pillar_2d_tmp}', fontsize=14)
    plt.gcf().autofmt_xdate()
    # axes[i].xaxis.set_visible(False)
    ax.set_xlim([0, int(10 * math.ceil(max(x) / 10.0))])
    ax.yaxis.set_tick_params(labelsize=11)
    # axes[i].axvline(x=0.5)
    sns.barplot(y=y, x=x, color="#0077b6").set(
        xlabel=None
    )  # colors_from_values(y, "rocket_r"))
    plt.subplots_adjust(hspace=0.5)
    plt.xlabel("proportion of excerpts containing tag (%)", fontsize=14)

    fig.suptitle(f"{col_name} proportions for each separate tag", fontsize=18)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, fontsize=12, loc=4, bbox_to_anchor=(0.9, 0))
    plt.savefig(
        f"Turkey_first_analysis_data/vizu/project_proportions/{col_name}.png",
        bbox_inches="tight",
        facecolor="white",
        dpi=200,
    )
    plt.show()


def get_vizu_2_lvl(col: pd.Series, col_name: str, ordered_level0: Dict[int, str]):

    whole_tag = sorted(flatten(col.apply(custom_eval)))
    tot_n_entries = len(col)
    level0 = [item.split("->")[0] for item in whole_tag]
    level1 = [item.split("->")[1] for item in whole_tag]
    n_level0 = len(set(level0))

    fig, axes = plt.subplots(
        n_level0, 1, sharex=True, figsize=(22, 10), facecolor="white"
    )

    for j in range(n_level0):
        pillar_2d_tmp = ordered_level0[j]
        subpilars_tmp = [
            level1[i] for i in range(len(level0)) if level0[i] == pillar_2d_tmp
        ]

        pillar_counts = dict(Counter(subpilars_tmp))
        proportions = {
            tag_name: round(100 * tag_count / tot_n_entries, 1)
            for tag_name, tag_count in pillar_counts.items()
        }

        if pillar_2d_tmp == "age":
            proportions = {
                tag_name: proportions[tag_name]
                for tag_name in [
                    "Older Persons (60+ years old)",
                    "Adult (18 to 59 years old)",
                    "Children (5 to 17 years old)",
                    "Infants/Toddlers (<5 years old)",
                ]
            }

        y = list(proportions.keys())
        x = list(proportions.values())

        axes[j].set_title(f"{pillar_2d_tmp}", fontsize=14)
        plt.gcf().autofmt_xdate()
        # axes[i].xaxis.set_visible(False)
        axes[j].set_xlim([0, 30])
        axes[j].yaxis.set_tick_params(labelsize=11)
        # axes[i].axvline(x=0.5)
        sns.barplot(ax=axes[j], y=y, x=x, color="#0077b6").set(
            xlabel=None
        )  # colors_from_values(y, "rocket_r"))
        plt.subplots_adjust(hspace=0.5)
        plt.xlabel("proportion of excerpts containing tag (%)", fontsize=14)

    fig.suptitle(f"{col_name} proportions for each separate tag", fontsize=18)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, fontsize=12, loc=4, bbox_to_anchor=(0.9, 0))
    plt.savefig(
        f"Turkey_first_analysis_data/vizu/project_proportions/{col_name}.png",
        bbox_inches="tight",
        facecolor="white",
        dpi=200,
    )
    plt.show()
