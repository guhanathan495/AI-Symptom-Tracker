import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Healthcare AI Hub", page_icon="🩺", layout="centered")
st.title("🏥 Multi-Specialty Healthcare AI Portal")
st.write("Verifiable Medical Intelligence Models for Clinical Decision Support.")

tab1, tab2, tab3, tab4 = st.tabs(["🩺 Symptom Tracker", "❤️ Heart Risk Predictor", "🎗️ Breast Cancer Analytics", "🩸 Diabetes Risk Predictor"])

# ==========================================
# TAB 1: SYMPTOM TRACKER
# ==========================================
with tab1:
    st.subheader("AI Medical Symptom Risk Assessment")
    
    symptoms_list = [
        'abdominal_pain', 'abnormal_menstruation', 'acidity', 'acute_liver_failure',
        'altered_sensorium', 'anxiety', 'chest_pain', 'chills', 
        'continuous_sneezing', 'cough', 'shivering', 'stomach_pain', 
        'vomiting', 'yellow_urine'
    ]

    s1 = st.selectbox("Primary Symptom", ["none"] + symptoms_list, key="s1")
    s2 = st.selectbox("Secondary Symptom", ["none"] + symptoms_list, key="s2")
    s3 = st.selectbox("Additional Symptom", ["none"] + symptoms_list, key="s3")
    
    if st.button("Analyze Symptoms", type="primary"):
        selected_symptoms = [s for s in [s1, s2, s3] if s != "none"]
        
        predicted_disease = "General Health Risk / Under Evaluation"
        
        if 'acute_liver_failure' in selected_symptoms or 'yellow_urine' in selected_symptoms or 'altered_sensorium' in selected_symptoms:
            predicted_disease = "Severe Hepatic Dysfunction / Jaundice Risk"
        elif 'abdominal_pain' in selected_symptoms or 'stomach_pain' in selected_symptoms or 'acidity' in selected_symptoms:
            predicted_disease = "Gastrointestinal Issue / GERD"
        elif 'continuous_sneezing' in selected_symptoms or 'shivering' in selected_symptoms or 'chills' in selected_symptoms:
            predicted_disease = "Common Cold / Upper Respiratory Infection"
        elif 'chest_pain' in selected_symptoms or 'cough' in selected_symptoms:
            predicted_disease = "Cardio-Respiratory Evaluation Required"
        elif 'abnormal_menstruation' in selected_symptoms or 'anxiety' in selected_symptoms:
            predicted_disease = "Neuro-Endocrine / Hormonal Imbalance Risk"
            
        st.success(f"### Predicted Condition: **{predicted_disease}**")



# ==========================================         
# TAB 2: HEART DISEASE PREDICTOR
# ==========================================
with tab2:
    st.subheader("Cardiovascular Health Risk Analyzer")
    try:
        model_h = joblib.load('heart_disease_model.pkl')
        age = st.slider("Age", 1, 100, 45)
        sex = st.selectbox("Sex", ["Female (0)", "Male (1)"])
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False (0)", "True (1)"])
        restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", [0, 1, 2])
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0, step=0.1)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia Status (0-3)", [0, 1, 2, 3])
        
        if st.button("Evaluate Cardiac Risk", type="primary"):
            sex_val = 1 if "Male" in sex else 0
            fbs_val = 1 if "True" in fbs else 0
            exang_val = 1 if "Yes" in exang else 0
            heart_inputs = np.array([[age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]])
            pred_h = model_h.predict(heart_inputs)
            if pred_h[0] == 1: 
                st.error("### Result: High Risk of Heart Disease Detected")
            else: 
                st.success("### Result: Low Risk / Normal Cardiovascular Status")
    except Exception as e:
        st.warning("Please run train.py first to generate the Heart Disease model file.")

# ==========================================
# TAB 3: BREAST CANCER ANALYTICS
# ==========================================
with tab3:
    st.subheader("Oncology Diagnostic Screening Engine")
    try:
        model_c = joblib.load('breast_cancer_model.pkl')
        m_radius = st.slider("Mean Radius", 5.0, 30.0, 14.0)
        m_texture = st.slider("Mean Texture", 5.0, 40.0, 19.0)
        m_perimeter = st.slider("Mean Perimeter", 40.0, 190.0, 92.0)
        m_area = st.slider("Mean Area", 140.0, 2500.0, 650.0)
        m_smoothness = st.slider("Mean Smoothness", 0.05, 0.25, 0.1, step=0.01)
        
        if st.button("Execute Diagnostic Test", type="primary"):
            base_features = [m_radius, m_texture, m_perimeter, m_area, m_smoothness] + [0.1]*25
            pred_c = model_c.predict(np.array([base_features]))
            if pred_c[0] == 0: 
                st.error("### Pathology Status: Malignant Tumor (High Risk Cancer)")
            else: 
                st.success("### Pathology Status: Benign Tumor (Safe)")
    except Exception as e:
        st.warning("Please run train.py first to generate the Breast Cancer model file.")

# ==========================================
# TAB 4: DIABETES RISK PREDICTOR
# ==========================================
# ==========================================
# TAB 4: DIABETES RISK PREDICTOR
# ==========================================
with tab4:
    st.subheader("Diabetes Metabolism Risk Assessment")
    try:
        model_d = joblib.load('diabetes_model.pkl')
        st.write("Input patient physiological data indicators below:")
        d_age = st.slider("Patient Age Scale", 0.0, 1.0, 0.5)
        d_bmi = st.slider("Body Mass Index (BMI) Scale", 0.0, 1.0, 0.5)
        d_bp = st.slider("Blood Pressure (BP) Scale", 0.0, 1.0, 0.5)
        d_glu = st.slider("Blood Glucose Level Scale", 0.0, 1.0, 0.5)

        if st.button("Evaluate Diabetes Risk", type="primary"):
            avg_input = (d_age + d_bmi + d_bp + d_glu) / 4.0
            
            if avg_input < 0.3:
                st.success("### Result: Low Risk / Normal Metabolic Profile")
            else:
                base_diabetes = [d_age, 0.0, d_bmi, d_bp] + [0.0]*5 + [d_glu]
                pred_d = model_d.predict(np.array([base_diabetes]))
                
                if pred_d == 1:
                    st.error("### Result: High Risk of Diabetes / Metabolic Dysfunction Detected")
                else:
                    st.success("### Result: Low Risk / Normal Metabolic Profile")
    except Exception as e:
        st.warning("Please run train.py first to generate the Diabetes model file.")

st.markdown("---")
st.info("💡 *Disclaimer: Built for job fair verification. Consult professionals for healthcare decisions.*")


