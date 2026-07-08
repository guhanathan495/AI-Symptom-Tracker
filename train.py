import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import joblib

# 1. MODEL 1: Disease Symptom Tracker
df_symptom = pd.read_csv('DiseaseAndSymptoms.csv')
df_symptom = df_symptom.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df_symptom = df_symptom.fillna('none')
y_symptom = df_symptom['Disease']
X_symptom = pd.get_dummies(df_symptom.drop(columns=['Disease']))
joblib.dump(list(X_symptom.columns), 'model_columns.pkl')
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_symptom, y_symptom, test_size=0.2, random_state=42)
model_symptom = RandomForestClassifier(n_estimators=100, random_state=42)
model_symptom.fit(X_train_s, y_train_s)
joblib.dump(model_symptom, 'symptom_tracker_model.pkl')
print("Symptom Model Trained Successfully!")

# 2. MODEL 2: Heart Disease Predictor
df_heart = pd.read_csv('heart.csv')
y_heart = df_heart['target']
X_heart = df_heart.drop(columns=['target'])
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
model_heart = RandomForestClassifier(n_estimators=100, random_state=42)
model_heart.fit(X_train_h, y_train_h)
joblib.dump(model_heart, 'heart_disease_model.pkl')
print("Heart Model Trained Successfully!")

# 3. MODEL 3: Breast Cancer Classifier
cancer_data = load_breast_cancer()
X_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y_cancer = cancer_data.target
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)
model_cancer = RandomForestClassifier(n_estimators=100, random_state=42)
model_cancer.fit(X_train_c, y_train_c)
joblib.dump(model_cancer, 'breast_cancer_model.pkl')
print("Cancer Model Trained Successfully!")
print("All 3 Models Ready!")
