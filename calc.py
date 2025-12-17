import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
import pickle
import uuid
from datetime import datetime
import requests
from redcap import project
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Model
MODEL_FILE = 'xgb_model.pkl'
FEATURE_LIST = 'feature_list.pkl'
SHORT_NAMES = 'short_names.pkl'
URL = 'https://webhook.site/6f0f9f6d-6b4c-443a-b9ce-b3cbcff99b2b'

# Pycap database setup
api_url = 'https://redcap.smh.ca/redcap/api/'
# api_key = os.getenv('REDCAP_API_KEY')  # Read from environment 

# if not api_key:
#     st.error("REDCAP_API_KEY not found in .env file")
#     st.stop()# project = Project(api_url, api_key)

with open(MODEL_FILE, "rb") as f:
    clf = pickle.load(f)

with open(FEATURE_LIST, "rb") as f:
    features = pickle.load(f)

with open(SHORT_NAMES, "rb") as f:
    short_names = pickle.load(f)

diag_idx_list = [i for i,x in enumerate(features) if 'diag_' in x]

diag_list = [features[i] for i in diag_idx_list]
diag_short_list = [short_names[i] for i in diag_idx_list]

st.title("Onset Delirium Risk Prediction Calculator")
# st.subheader("This calculator predicts whether a patient is at risk of developing onset delirium.")

with st.sidebar:
    st.write("Delirium is an acute and fluctuating disturbance in attention and cognition. It is common among hospitalized older adults and is associated with increased morbidity, mortality, length of hospital stay, and healthcare costs. Early identification of patients at risk for delirium can help implement preventive strategies and improve patient outcomes.")
    st.write("This calculator uses a machine learning model to predict the risk of onset delirium based on patient characteristics and laboratory results.")
col1, col2 = st.columns(2)

import streamlit as st
# initialize only once
if "age_numeric" not in st.session_state:
    st.session_state.age_numeric = 60
if "age_slider" not in st.session_state:
    st.session_state.age_slider = 60
if "sodium_numeric" not in st.session_state:
    st.session_state.sodium_numeric = 60.0
if "sodium_slider" not in st.session_state:
    st.session_state.sodium_slider = 60.0
if "bilirubin_numeric" not in st.session_state:
    st.session_state.bilirubin_numeric = 1.0
if "bilirubin_slider" not in st.session_state:
    st.session_state.bilirubin_slider = 1.0

def update_age_slider():
    st.session_state.age_slider = st.session_state.age_numeric
def update_numin():
    st.session_state.age_numeric = st.session_state.age_slider

def update_sodium_slider():
    st.session_state.sodium_slider = st.session_state.sodium_numeric
def update_sodium_numeric():
    st.session_state.sodium_numeric = st.session_state.sodium_slider


def update_bilirubin_slider():
    st.session_state.bilirubin_slider = st.session_state.bilirubin_numeric
def update_bilirubin_numeric():
    st.session_state.bilirubin_numeric = st.session_state.bilirubin_slider


# Form for user inputs
with col1:
    user_inputs = {}
    st.markdown("### Calculator Inputs")
    mrn = st.number_input("MRN", min_value=0, max_value=99999999, key='mrn', help="Enter the patient's Medical Record Number (MRN).")
    st.write("Enter values for the features below:")
    age_cols_left, age_cols_right = st.columns([.3,.7])
    with age_cols_left:
        user_inputs['Age'] = st.number_input("Age",
                                             min_value=18,
                                             max_value=120,
                                            #  value=60,
                                             key='age_numeric',
                                             on_change=update_age_slider,
                                             
                                             )
    with age_cols_right:
        user_inputs['Age'] = st.slider("Age", 
                                       min_value=18, 
                                       max_value=120, 
                                    #    value=60,
                                       key='age_slider',
                                       on_change=update_numin,
                                       label_visibility='hidden',
                                       help = "Enter the patient's age in years.")

    sodium_cols_left, sodium_cols_right = st.columns([.3,.7])

    with sodium_cols_left:
        user_inputs['Sodium (Moles/volume)'] = st.number_input("Sodium (Moles/volume)", min_value=0.0, max_value=200.0,
                                                               key='sodium_numeric', on_change=update_sodium_slider)
    with sodium_cols_right:
        user_inputs['Sodium (Moles/volume)'] = st.slider("Sodium (Moles/volume)", min_value=0.0, max_value=200.0,
                                                          key='sodium_slider', on_change=update_sodium_numeric,
                                                          label_visibility='hidden',
                                                          help="Enter the patient's sodium level in moles per volume.")

    bilirubin_cols_left, bilirubin_cols_right = st.columns([.3,.7])

    with bilirubin_cols_left:
        user_inputs['Bilirubin (Moles/volume)'] = st.number_input("Bilirubin (Moles/volume)", min_value=0.0, max_value=2.0,
                                                              key='bilirubin_numeric', on_change=update_bilirubin_slider)
    with bilirubin_cols_right:
        user_inputs['Bilirubin (Moles/volume)'] = st.slider("Bilirubin (Moles/volume)", min_value=0.0, max_value=2.0,
                                                          key='bilirubin_slider', on_change=update_bilirubin_numeric,
                                                          label_visibility='hidden',
                                                          help="Enter the patient's bilirubin level in moles per volume.")

    diagnoses = st.multiselect("Select Diagnoses",
                                options=diag_short_list,
                                help="Select all diagnoses that apply to the patient.")
    # Submit button
    submitted = st.button("Run Model")
    with col2:
        if submitted:
            if mrn is None or mrn == 0:
                st.error("Please enter an MRN.")
            elif user_inputs['Age'] is None or user_inputs['Age'] == 0:
                st.error("Please enter a valid age (18 or older).")
            else:
                unique_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                for diag in diag_short_list:
                    if diag in diagnoses:
                        user_inputs[f'{diag}'] = 1
                    else:
                        user_inputs[f'{diag}'] = 0


                df = pd.DataFrame([user_inputs])
                df = df[short_names]
                df.columns = features

                st.markdown("### Model Output")
                
                # Prepare input for the model
                # pred = clf.predict(df)[0]
                pred_proba = clf.predict_proba(df)[:, 1][0]
                if pred_proba < 0.5:
                    st.markdown('## :green[Low Risk]')
                elif 0.5 <= pred_proba < 0.75:
                    st.markdown('## :orange[Moderate Risk]')
                elif pred_proba >= 0.75:
                    st.markdown('## :red[High Risk]')

                st.info(f"**Unique Identifier:** `{unique_id}`")
                st.info(f"**Prediction generated at** `{timestamp}`")

                dict_ = df.iloc[0].to_dict()
                dict_['unique_identifier'] = unique_id
                dict_['timestamp'] = timestamp
                dict_['pred_proba'] = pred_proba
                dict_['mrn'] = mrn
                for k,v in dict_.items():
                    st.info(k)
                    if isinstance(v, (np.integer, np.floating)):
                        dict_[k] = v.item()

                # Push data to redcap
                to_import = [dict]
                # response = project.import_records(to_import)
               
                x = requests.post(URL, json = dict_)
                if x.status_code != 200:
                    st.error("Error saving results to the database.")
    
