from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import operator


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


def get_barplots(col: pd.Series, col_name: str, dir_name: str):
    if "->" in str(col):
        get_vizu_2_lvl(col, col_name, dir_name)
    else:
        get_vizu_1_lvl(col, col_name, dir_name)


def get_vizu_1_lvl(col: pd.Series, col_name: str, dir_name):

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
        dir_name,
        bbox_inches="tight",
        facecolor="white",
        dpi=200,
    )
    plt.show()


def get_vizu_2_lvl(col: pd.Series, col_name: str, dir_name: str):
    """
    vizu for hierarchial tags (example: pillars->subpillars)
    """

    whole_tag = sorted(flatten(col.apply(custom_eval)))
    tot_n_entries = len(col)
    level0 = [item.split("->")[0] for item in whole_tag]
    n_level0 = len(set(level0))

    tot_subpillar_counts = dict(Counter(whole_tag))
    tot_proportions = {
        tag_name: round(100 * tag_count / tot_n_entries, 1)
        for tag_name, tag_count in tot_subpillar_counts.items()
    }
    max_prop = max(list(tot_proportions.values()))
    unique_level1 = list(tot_proportions.keys())
    unique_level0 = [item.split("->")[0] for item in unique_level1]

    level0_counts = dict(Counter(unique_level0))

    sorted_d = dict(
        sorted(level0_counts.items(), key=operator.itemgetter(1), reverse=True)
    )
    level0_ordering = list(sorted_d.keys())

    fig, axes = plt.subplots(
        n_level0,
        1,
        sharex=True,
        figsize=(10 + n_level0, 3 + n_level0 * 2),
        facecolor="white",
    )

    for j in range(n_level0):
        level0_tmp = level0_ordering[j]

        proportions = {
            tag_name.split("->")[1]: tag_count
            for tag_name, tag_count in tot_proportions.items()
            if level0_tmp in tag_name
        }

        if level0_tmp == "age":
            age_proportions = {}
            for name in ["old", "adult", "child", "infant"]:
                age_proportions.update(
                    {
                        tag_name: tag_count
                        for tag_name, tag_count in tot_proportions.items()
                        if name in tag_name.lower()
                    }
                )

            proportions = age_proportions

        y = list(proportions.keys())
        x = list(proportions.values())

        axes[j].set_title(f"{level0_tmp}", fontsize=14)
        plt.gcf().autofmt_xdate()
        # axes[i].xaxis.set_visible(False)
        axes[j].set_xlim([0, max_prop + max_prop // 5])
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
        dir_name,
        bbox_inches="tight",
        facecolor="white",
        dpi=200,
    )
    plt.show()
