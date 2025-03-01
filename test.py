# -*- coding: utf-8 -*-
"""
Disease prediction model using XGBoost
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Global variables to store model and encoder
global_model = None
global_label_encoder = None

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Disease'])  # Features
    y = df['Disease']  # Target
    # Encode target labels (Disease)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded, label_encoder

# Train the Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(set(y)), eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Training Completed. Accuracy: {accuracy * 100:.2f}%")
    return model

# Predict Disease
def predict_disease(model, label_encoder, user_input):
    user_data = np.array([user_input])  # Convert to NumPy array
    prediction = model.predict(user_data)
    predicted_disease = label_encoder.inverse_transform(prediction)[0]
    return predicted_disease

# Initialize model - can be called from external code
def initialize_model(file_path):
    global global_model, global_label_encoder
    X, y, global_label_encoder = load_data(file_path)
    global_model = train_model(X, y)
    return global_model, global_label_encoder

# Get prediction using global model - for convenience
def get_prediction(user_input):
    if global_model is None or global_label_encoder is None:
        raise ValueError("Model not initialized. Call initialize_model first.")
    return predict_disease(global_model, global_label_encoder, user_input)

# Main execution (this runs only if the script is executed directly)
if __name__ == "__main__":
    FILE_PATH = "/content/symptom_disease_dataset.csv"
    X, y, label_encoder = load_data(FILE_PATH)
    model = train_model(X, y)
    
    # For testing in standalone mode, you could add code here
    # to get user input via console and make predictions
    print("\nNow, input symptoms to predict the disease:")
    # Example of test prediction with dummy data
    test_input = [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0]  # Sample symptoms
    predicted = predict_disease(model, label_encoder, test_input)
    print(f"\n✅ Predicted Disease: **{predicted}**")