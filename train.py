import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the dataset
# Make sure the CSV file name matches exactly
df = pd.read_csv('DiseaseAndSymptoms.csv')

# Strip any whitespace from text columns to prevent formatting issues
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Replace missing values (NaN) with 'none'
df = df.fillna('none')

# 2. Split into Features (X) and Target (y)
y = df['Disease']
X_raw = df.drop(columns=['Disease'])

# 3. Convert text features to numbers using One-Hot Encoding
X = pd.get_dummies(X_raw)

# Save the column structures for future web deployment alignment
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')

# 4. Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Training Success! Accuracy: {accuracy * 100:.2f}%")

# 7. Save the trained model to a file for deployment
joblib.dump(model, 'symptom_tracker_model.pkl')
print("Model saved successfully as 'symptom_tracker_model.pkl'")
