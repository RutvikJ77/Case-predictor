"""

"""

from pandas.core.frame import DataFrame
import streamlit as st
import logging
from processing import infer_only_meta
import pandas as pd
import glob
import tensorflow as tf

st.set_page_config(
   page_title="Case Predictor🔎",
   page_icon="🔎",
   layout="centered",
   menu_items={
      'About': "#### Get a predicted rate of infection easily!",
      'Report a bug': "https://github.com/RutvikJ77/Case-predictor",
    }
)

models=[]
for i in glob.glob(f'./models/*'):
        models.append(tf.keras.models.load_model(i))

def analyse(user_upload):
    results, data_frame = infer_only_meta(pd.read_csv(user_upload), models, 100)
    return data_frame

def file_check(uploaded_file):
    if uploaded_file is not None:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}

            if file_details["FileType"]!="text/csv":
                st.write("Please enter correct file type.")

def convert_df(df):
    return df.to_csv().encode('utf-8')         


st.title("Case predictor🔎")

html = """ 
<style>
    .element-container {
      padding:20px;
    }
</style>

"""

col1, col2 = st.columns((3,1))

user_upload = col1.file_uploader("Upload the observation file:", type=["csv"])
file_check(user_upload)

col2.markdown(html, unsafe_allow_html=True)

if col2.button("Analyse"):
    logging.info("Analyse called.")
    data_frame = analyse(user_upload)
    st.line_chart(data_frame)
    st.download_button(label="Download", data=convert_df(data_frame), file_name="predictions.csv", mime='text/csv')