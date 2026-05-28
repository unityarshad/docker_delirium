import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
import pickle
import uuid
from datetime import datetime
import requests
from redcap import Project
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Model
MODEL_FILE = 'final_10_feature_xgb_model_2026_05_25.pkl'
FEATURE_LIST = 'feature_list.pkl'
SHORT_NAMES = 'short_names.pkl'

# Pycap database setup
api_url = 'https://redcap.smh.ca/redcap/api/'
api_key = os.getenv('REDCAP_API_KEY')  # Read from environment 

if not api_key:
    st.error("REDCAP_API_KEY not found in .env file")
    st.stop()
    
project = Project(api_url, api_key)

with open(MODEL_FILE, "rb") as f:
    clf = pickle.load(f)

with open(FEATURE_LIST, "rb") as f:
    features = pickle.load(f)

with open(SHORT_NAMES, "rb") as f:
    short_names = pickle.load(f)

st.title("Onset Delirium Risk Prediction Calculator")
# st.subheader("This calculator predicts whether a patient is at risk of developing onset delirium.")
st.subheader("Please only refer to patient's documentation for this current admission. This includes reviewing the ED physician note, medicine physician admission note/initial consultation note and lab test results.")

with st.sidebar:
    st.write("Delirium is an acute and fluctuating disturbance in attention and cognition. It is common among hospitalized older adults and is associated with increased morbidity, mortality, length of hospital stay, and healthcare costs. Early identification of patients at risk for delirium can help implement preventive strategies and improve patient outcomes.")
    st.write("This calculator uses a machine learning model to predict the risk of onset delirium based on patient characteristics and laboratory results.")
col1, col2 = st.columns(2)

# initialize only once
if "age_numeric" not in st.session_state:
    st.session_state.age_numeric = 60
if "age_slider" not in st.session_state:
    st.session_state.age_slider = 60
if "albumin_numeric" not in st.session_state:
    st.session_state.albumin_numeric = 34.71
if "albumin_slider" not in st.session_state:
    st.session_state.albumin_slider = 34.71


def update_age_slider():
    st.session_state.age_slider = st.session_state.age_numeric
def update_numin():
    st.session_state.age_numeric = st.session_state.age_slider

def update_albumin_slider():
    st.session_state.albumin_slider = st.session_state.albumin_numeric
def update_albumin_numeric():
    st.session_state.albumin_numeric = st.session_state.albumin_slider

# Form for user inputs
with col1:
    user_inputs = {}
    st.markdown("### Calculator Inputs")
    mrn = st.number_input("MRN", min_value=0, max_value=99999999, key='mrn', help="Enter the patient's Medical Record Number (MRN).", value = None)
    hospitals = ["HRH", "LHSC-U", "LHSC-V", "Niagara", "NYGH", "SBK", "SHN-B", "SHN-G", "SJHC", "SMH", "THP-CV", "THP-M", "TWH"]  # Replace with your list of hospitals
    selected_hospital = st.selectbox("Select Hospital", hospitals)
    user_inputs['Hospital'] = selected_hospital
    st.write("Enter values for the features below:")
    age_cols_left, age_cols_right = st.columns([.3,.7])
    with age_cols_left:
        user_inputs['age'] = st.number_input("Age",
                                             min_value=18,
                                             max_value=120,
                                            #  value=60,
                                             key='age_numeric',
                                             on_change=update_age_slider,
                                             
                                             )
    with age_cols_right:
        user_inputs['age'] = st.slider("Age", 
                                       min_value=18, 
                                       max_value=120, 
                                    #    value=60,
                                       key='age_slider',
                                       on_change=update_numin,
                                       label_visibility='hidden',
                                       help = "Enter the patient's age in years.")

    sex = st.selectbox("Select Sex", options=['Male', 'Female'], help="Select the patient's sex.")
    if sex == 'Male':
        user_inputs['Gender'] = 0
    else:
        user_inputs['Gender'] = 1
        

    albumin_cols_left, albumin_cols_right = st.columns([.3,.7])

    with albumin_cols_left:
        user_inputs['lab_albumin'] = st.number_input("Albumin (Mass/volume) Value for the FIRST RESULT of this encounter", min_value=10.0, max_value=65.0, 
                                                               key='albumin_numeric', on_change=update_albumin_slider)
    with albumin_cols_right:
        user_inputs['lab_albumin'] = st.slider("Albumin (Mass/volume)", min_value=10.0, max_value=65.0,
                                                          key='albumin_slider', on_change=update_albumin_numeric,
                                                          label_visibility='hidden',
                                                          help="Enter the patient's albumin level in mass per volume.")

    st.markdown("#### Patient Diagnoses")
    st.caption("Please review and check all the current or historical diagnoses that apply. Unchecked diagnoses default to 0.")


    diag_features = [x for x in features if 'diag_' in x]
    diag_short_names = [short_names[i] for i, feat in enumerate(features) if 'diag_' in feat]

    # Split layout into 2 columns for better spatial design
    chk_col1, chk_col2 = st.columns(2)
    for idx, diag in enumerate(diag_short_names):
        # Alternate rendering between the two columns
        with chk_col1 if idx % 2 == 0 else chk_col2:
            real_name = diag_features[idx]
            checked = st.checkbox(diag, value=False)
            user_inputs[real_name] = 1 if checked else 0

    # Submit button
    submitted = st.button("Run Model")
    with col2:
        if submitted:
            if mrn is None or mrn == 0:
                st.error("Please enter an MRN.")
            elif user_inputs['age'] is None or user_inputs['age'] == 0:
                st.error("Please enter a valid age (18 or older).")
            else:
                unique_id = "calc_" + str(uuid.uuid4()).split('-')[0]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                df = pd.DataFrame([user_inputs])
                
                df = df[features]
                df.columns = features

                st.markdown("### Model Output")
                
                # Prepare input for the model
                # pred = clf.predict(df)[0]
                pos_predict = False
                pred_proba = float(clf.predict_proba(df)[:, 1][0])
                if pred_proba < 0.45:
                    st.markdown('### :green[Low Risk]')
                elif 0.45 <= pred_proba < 0.75:
                    pos_predict = True
                    st.markdown('### :orange[Moderate Risk]')
                elif pred_proba >= 0.75:
                    pos_predict = True
                    st.markdown('### :red[High Risk]')
                

                st.info(f"**Unique Identifier:** `{unique_id}`")
                st.info(f"**Prediction generated at** `{timestamp}`")

                model_dict = df.iloc[0].to_dict()
                for k, v in model_dict.items():
                    if isinstance(v, (np.integer, np.floating)):
                        model_dict[k] = v.item()

                # Hospital Mapping
                hospital_mapping = {name: idx + 1 for idx, name in enumerate(hospitals)}
                mapped_hospital_id = hospital_mapping.get(selected_hospital)

                # mrn_fields form
                base_record = {
                    'record_id': unique_id,
                    'unique_id_mrn': unique_id,
                    'mrn': mrn
                }

                # delirium_input form
                repeating_record = {
                    'record_id': unique_id,
                    'redcap_repeat_instrument': 'delirium_input',
                    'redcap_repeat_instance': 'new',
                    'hospital_id': mapped_hospital_id,
                    'unique_id_del_input': unique_id,
                    'timestamp': timestamp,
                    'pred_proba': pred_proba,
                    'gender_f': str(user_inputs['Gender'])
                }

                # Copy features from the model inputs into the repeating record instrument
                for k, v in model_dict.items():
                        repeating_record[k] = v

                # Troubleshooting output
                # for key, val in base_record.items():
                #     st.info(f'[Base Form] {key}: {val}')
                # for key, val in repeating_record.items():
                #     st.info(f'[Repeating Form] {key}: {val}')

                # Push split data to REDCap
                to_import = [base_record, repeating_record]
                try:
                    response = project.import_records(to_import)
                    st.success(f"Successfully uploaded to REDCap!") 
                    # st.success(f"Response: {response}")
                except Exception as e:
                    st.error(f"REDCAP Error: {e}. Ensure your dictionary keys match your REDCap field names exactly.")
                    
