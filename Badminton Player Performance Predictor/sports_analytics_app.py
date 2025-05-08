import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression

# Load saved Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Load or initialize Linear Regression model (for demonstration, train dummy one)
# In a real case, load your trained Linear Regression model as well
lr_model = LinearRegression()
lr_model.fit([[0]*8], [0])  # Dummy fit, replace with your real model

# Feature list
features = [
    'age',
    'height_cm',
    'weight_kg',
    'bmi',
    'training_hours_per_week',
    'matches_played_last_month',
    'injury_history',
    'recovery_days_last_injury'
]

# Streamlit app
st.title("🏋️ Sports Analytics: Injury/Performance Prediction")
st.write("This app predicts outcomes using Random Forest and Linear Regression models based on player information.")

# Option to upload CSV
upload_option = st.radio("Choose input method:", ("Manual Input", "Upload CSV"))

if upload_option == "Manual Input":
    st.subheader("Enter Player Info Manually")
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", value=0.0)

    input_df = pd.DataFrame([user_input])

elif upload_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# Prediction Section
if 'input_df' in locals():
    st.subheader("📊 Predictions")

    rf_preds = rf_model.predict(input_df)
    lr_preds = lr_model.predict(input_df)

    input_df['Random Forest Prediction'] = rf_preds
    input_df['Linear Regression Prediction'] = lr_preds

    st.write("### Prediction Output:")
    st.dataframe(input_df)

    # Optional: Add visuals or performance comparison later
    st.success("Prediction completed!")
