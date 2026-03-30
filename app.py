import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import shap

# Title
st.title("Customer Churn Prediction with Explainable AI")

# Load dataset
df = pd.read_csv(r"C:\Users\mudhi\OneDrive\Desktop\IOMP\Telco-Customer-Churn.csv")

# Preprocessing
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# UI Inputs (simple few features)
st.sidebar.header("Enter Customer Details")
# New Behavioral Inputs
login_time = st.sidebar.slider("Login Duration (minutes)", 0, 300, 30)
inactive_time = st.sidebar.slider("Inactive Time (minutes)", 0, 300, 50)
weekly_visits = st.sidebar.slider("Weekly Visits", 0, 50, 5)
tenure = st.sidebar.slider("Tenure", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0, 150, 70)
total = st.sidebar.slider("Total Charges", 0, 10000, 2000)

# Create input (simplified)
input_data = X.iloc[0:1].copy()
# Add new features (temporary simulation)
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly
input_data['TotalCharges'] = total

# Add behavioral features (IMPORTANT)
input_data['tenure'] += weekly_visits   # simulate engagement
input_data['MonthlyCharges'] += login_time * 0.1
input_data['TotalCharges'] -= inactive_time * 0.5

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer is NOT likely to churn")

    # SHAP explanation
    explainer = shap.Explainer(model, X)
    shap_values = explainer(input_data)

    st.subheader("Feature Impact (Explainability)")
    st.write(shap_values.values[0][:, 1])