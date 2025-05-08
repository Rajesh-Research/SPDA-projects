
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor

st.set_page_config(page_title="ODI Batsman Runs Predictor", layout="centered")

st.title("🏏 Predicting Batsman Runs in ODI Matches")
st.markdown("""
This app predicts how many runs a batsman might score in an upcoming ODI match using machine learning.

### Instructions:
- Fill in the batsman, teams, and match details below.
- The model will return a predicted number of runs.

*Note: The model is trained on historical data and includes features like form, venue, batting order, and match pressure.*
""")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("batsman_run_predictor.pkl")

model = load_model()

# Load encoders and feature layout
@st.cache_data
def load_metadata():
    return joblib.load("model_metadata.pkl")

metadata = load_metadata()
categorical_features = metadata['categorical_features']
feature_columns = metadata['feature_columns']

# Input UI
st.subheader("Match Input Details")

user_input = {}
user_input['batsman'] = st.selectbox("Batsman", metadata['batsman_list'])
user_input['batting_team'] = st.selectbox("Batting Team", metadata['team_list'])
user_input['bowling_team'] = st.selectbox("Bowling Team", metadata['team_list'])
user_input['venue'] = st.selectbox("Venue", metadata['venue_list'])
user_input['city'] = st.selectbox("City", metadata['city_list'])
user_input['toss_winner'] = st.selectbox("Toss Winner", metadata['team_list'])
user_input['toss_decision'] = st.selectbox("Toss Decision", ['bat', 'field'])

user_input['season'] = st.number_input("Season (e.g., 2023)", min_value=2002, max_value=2025, value=2023)
user_input['form_avg_3'] = st.number_input("Batsman's average (last 3 matches)", min_value=0.0, value=25.0)
user_input['form_avg_5'] = st.number_input("Batsman's average (last 5 matches)", min_value=0.0, value=27.5)
user_input['batting_position'] = st.slider("Batting Position", 1, 11, value=3)
user_input['innings'] = st.radio("Innings", [1, 2])
user_input['pressure_score'] = st.number_input("Estimated Pressure Score (based on target)", value=250)

# Build dataframe from user input
input_df = pd.DataFrame([user_input])

# One-hot encode
input_encoded = pd.get_dummies(input_df, columns=categorical_features)
for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0  # Add missing columns

input_encoded = input_encoded[feature_columns]  # Ensure column order

# Predict
if st.button("Predict Runs"):
    predicted_runs = model.predict(input_encoded)[0]
    st.success(f"🎯 Predicted Runs: {round(predicted_runs, 2)}")
