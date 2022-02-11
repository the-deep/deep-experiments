import pandas as pd
import streamlit as st


st.set_page_config(page_title='Data Dashboard', page_icon="./assets/images/icon.deep.png",
                   layout='wide', initial_sidebar_state='auto')

path = "val_v0.7.1.csv" #st.text_input("CSV Path", value="", key="txt_inp_path")
#display_csv = st.button("Read CSV", key="btn_display_csv")
display_csv = True

if display_csv:
    df = pd.read_csv(path)
    df['excerpt'] = df['excerpt'].str.wrap(150) 
    keys = df.columns.tolist()
    on_keys = st.multiselect("Columns", keys, keys)
    df = df[on_keys]
    df = df[:1000]

    def hover(hover_color="#ffff99"):
        return dict(
            selector="tr:hover",
            props=[("background-color", "%s" % hover_color)],
        )


    styles = [
        hover(),
        dict(
            selector="th",
            props=[("font-size", "150%"), ("text-align", "center")],
        ),
        dict(selector="caption", props=[("caption-side", "bottom")]),
    ]

    # Table view. Use pands styling.
    style = df.style.set_properties(
        **{"text-align": "left", "white-space": "pre"}
    ).set_table_styles([dict(selector="th", props=[("text-align", "left")])])
    pd.set_option('display.max_colwidth', 0)
    style = style.set_table_styles(styles)
    st.table(style)
