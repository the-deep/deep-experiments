import os
import json
import streamlit.components.v1 as components

# Create a _RELEASE constant. We'll set this to False while we're developing
# the component, and True when we're ready to package and distribute it.
# (This is, of course, optional - there are innumerable ways to manage your
# release process.)
_RELEASE = False

# Declare a Streamlit component. `declare_component` returns a function
# that is used to create instances of the component. We're naming this
# function "_component_func", with an underscore prefix, because we don't want
# to expose it directly to users. Instead, we will create a custom wrapper
# function, below, that will serve as our component's public API.

# It's worth noting that this call to `declare_component` is the
# *only thing* you need to do to create the binding between Streamlit and
# your component frontend. Everything else we do in this file is simply a
# best practice.

if not _RELEASE:
    _component_func = components.declare_component(
        # We give the component a simple, descriptive name ("custom_table_component"
        # does not fit this bill, so please choose something better for your
        # own component :)
        "custom_table_component",
        # Pass `url` here to tell Streamlit that the component will be served
        # by the local dev server that you run via `npm run start`.
        # (This is useful while your component is in development.)
        url="http://localhost:3001",
    )
else:
    # When we're distributing a production version of the component, we'll
    # replace the `url` param with `path`, and point it to to the component's
    # build directory:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("custom_table_component", path=build_dir)


# Create a wrapper function for the component. This is an optional
# best practice - we could simply expose the component function returned by
# `declare_component` and call it done. The wrapper allows us to customize
# our component's API: we can pre-process its input args, post-process its
# output value, and add a docstring for users.
def custom_table_component(data, key=None):
    # Call through to our private component function. Arguments we pass here
    # will be sent to the frontend, where they'll be available in an "args"
    # dictionary.
    #
    # "default" is a special argument that specifies the initial return
    # value of the component before the user has interacted with it.
    component_value = _component_func(data=data, key=key, default=0)

    # We could modify the value returned from the component if we wanted.
    # There's no need to do this in our simple example - but it's an option.
    return component_value


# Add some test code to play with the component while it's in development.
# During development, we can run this just as we would any other Streamlit
# app: `$ streamlit run custom_table_component/__init__.py`
if not _RELEASE:
    import time
    from ast import literal_eval
    import boto3

    # import st_aggrid
    import pandas as pd
    import streamlit as st

    DATA_PATH = "data/test_environment/offline_testing/entries_vs_sentences_tags.csv"
    table = pd.read_csv(DATA_PATH)
    ##
    table.loc[table["Pillars"].isna(), "Pillars"] = "[]"
    table["Pillars"] = table["Pillars"].apply(literal_eval)
    table["Pillars"] = table["Pillars"].apply(lambda x: ", ".join(set(x)))
    table.loc[table["Sectors"].isna(), "Sectors"] = "[]"
    table["Sectors"] = table["Sectors"].apply(literal_eval)
    table["Sectors"] = table["Sectors"].apply(lambda x: ", ".join(set(x)))
    ##
    table.loc[table["Predicted Pillars"].isna(), "Predicted Pillars"] = "[]"
    table["Predicted Pillars"] = table["Predicted Pillars"].apply(literal_eval)
    table["Predicted Pillars"] = table["Predicted Pillars"].apply(lambda x: ", ".join(set(x)))
    table.loc[table["Predicted Sectors"].isna(), "Predicted Sectors"] = "[]"
    table["Predicted Sectors"] = table["Predicted Sectors"].apply(literal_eval)
    table["Predicted Sectors"] = table["Predicted Sectors"].apply(lambda x: ", ".join(set(x)))
    ##
    st.set_page_config(layout="wide")
    projects = table["Project Title"].unique().tolist()
    selected_project_title = st.selectbox("Select a Project", projects)
    selected_project = table[table["Project Title"].eq(selected_project_title)]

    af_title = selected_project["Analysis Framework Title"].iloc[0]
    st.write(f"Analytical Framework: **{af_title}**")
    lead_urls = selected_project["Lead URL"].unique().tolist()
    selected_url = st.selectbox("Select a Lead:", lead_urls)
    selected_lead = selected_project[selected_project["Lead URL"].eq(selected_url)]
    selected_lead.drop(
        columns=["Lead URL", "Project Title", "Analysis Framework Title"],
        inplace=True,
        axis=1,
    )
    st.write(f"[Selected Lead URL]({selected_url})")
    st.markdown(
        """
    <style>
        table tr:nth-child(even) {
            background-color: #f2f2f2;
        }

    table {
            width: 100%;
            overflow-x: auto;
        }
    table td:nth-child(1) {
        display: none
    }
    table th:nth-child(1) {
        display: none
    }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    cleaned_table = selected_lead.replace("", None)
    json_data = json.dumps(cleaned_table.to_dict("records"))
    clicked_text = custom_table_component(json_data)
    print(clicked_text)
    entry = "entry"

    def send_feed_back(feedback_text, lead_url, project_title, af_title):
        try:
            bucket_name = "deep-test-environment"
            s3_client = boto3.client("s3")
            model_identifier = "paraphrase-multilingual-mpnet-base-v2_secotrs-and-pillars"
            feedback_obj = {
                "feedback_text": feedback_text,
                "lead_url": lead_url,
                "project_title": project_title,
                "af_title": af_title,
                "model_identifier": model_identifier,
            }
            feedback_obj = json.dumps(feedback_obj).encode("utf-8")
            file_name_on_s3 = f"feedback/{time.ctime().replace(' ', '_')}.json"
            s3_client.put_object(Body=feedback_obj, Bucket=bucket_name, Key=file_name_on_s3)
            st.success("Feedback submitted successfully.")
        except Exception as e:
            st.warning(e.message)

    if clicked_text:
        send_feed_back(clicked_text, selected_url, selected_project_title, af_title)
