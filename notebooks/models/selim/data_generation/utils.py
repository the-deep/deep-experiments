import re
import pandas as pd


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
