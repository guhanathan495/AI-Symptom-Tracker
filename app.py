import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config for Mobile Optimization
st.set_page_config(page_title="AI Medical Symptom Tracker", page_icon="🩺", layout="centered")

# Load trained model and features tracking metrics
model = joblib.load('symptom_tracker_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("🩺 AI Medical Symptom Tracker")
st.write("Select your symptoms below to analyze potential health risks.")

st.markdown("---")

# UI Form Inputs for the Interviewer
st.subheader("Select Main Symptoms")

# Clean symptom names from columns for dropdown presentation
all_features = [col for col in model_columns]
symptoms_set = set()
for col in all_features:
    if '_' in col:
        # Split by underscore and get the actual symptom name
        parts = col.split('_')
        if len(parts) > 2:
            symptoms_set.add("_".join(parts[2:]))
        else:
            symptoms_set.add(parts[-1])

symptoms_list = sorted(list(symptoms_set))

# Dropdowns for User/HR interaction
symptom_1 = st.selectbox("Primary Symptom", ["none"] + symptoms_list)
symptom_2 = st.selectbox("Secondary Symptom", ["none"] + symptoms_list)
symptom_3 = st.selectbox("Additional Symptom", ["none"] + symptoms_list)

if st.button("Predict Health Risk Status", type="primary"):
    # Build empty schema dataframe mirroring model tracking layers
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Map the selected options back into system binary flags
    user_inputs = [symptom_1, symptom_2, symptom_3]
    for i, sym in enumerate(user_inputs):
        if sym != "none":
            # Check for matches in model columns
            col_name = f"Symptom_{i+1}_{sym}"
            if col_name in input_data.columns:
                input_data.loc[0, col_name] = 1
                
    # Execute Model Evaluation
    prediction = model.predict(input_data)
    
    st.markdown("---")
    st.success(f"### 📋 Predicted Health Condition: **{prediction[0]}**")
    st.info("💡 *Disclaimer: This is an AI clinical support prototype for job fair verification. Consult professionals for healthcare decisions.*")
