
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ODI Batsman Runs Predictor", layout="centered")

st.title("🏏 Predicting Batsman Runs in ODI Matches")
st.markdown("""
This app predicts how many runs a batsman might score in an upcoming ODI match using machine learning.

### Key Features:
- 🎯 Batsman dropdown filtered by Batting Team
- 🏟️ Venue auto-maps to City (locked)
- 📈 Pressure is inferred based on innings
- 🛠️ Toss Winner & Toss Decision removed for clarity

*Model trained on real ODI match data.*
""")

@st.cache_resource
def load_model():
    return joblib.load("batsman_run_predictor.pkl")

@st.cache_data
def load_metadata():
    return joblib.load("model_metadata.pkl")

model = load_model()
metadata = load_metadata()

team_batsmen = {
    "India": ["V Kohli", "RG Sharma"],
    "Australia": ["DA Warner"]
}

venue_city_map = {
    "Eden Gardens": "Kolkata",
    "Wankhede Stadium": "Mumbai",
    "MCG": "Melbourne"
}

st.subheader("Match Input Details")

batting_team = st.selectbox("Batting Team", metadata['team_list'])

if batting_team not in team_batsmen:
    st.error("❌ This team is not supported yet. Please pick a team with mapped batsmen.")
    st.stop()

batsman_list = team_batsmen[batting_team]
batsman = st.selectbox("Batsman", batsman_list)

bowling_team = st.selectbox("Bowling Team", [team for team in metadata['team_list'] if team != batting_team])

venue = st.selectbox("Venue", list(venue_city_map.keys()))
city = venue_city_map[venue]
st.text_input("City (auto-mapped from venue)", value=city, disabled=True)

season = st.number_input("Season (e.g., 2023)", min_value=2002, max_value=2025, value=2023)
form_avg_3 = st.slider("Batsman's average (last 3 matches)", min_value=1.0, max_value=150.0, value=25.0)
form_avg_5 = st.slider("Batsman's average (last 5 matches)", min_value=1.0, max_value=150.0, value=30.0)
batting_position = st.slider("Batting Position", 1, 11, value=3)
innings = st.radio("Innings", [1, 2])

if innings == 1:
    pressure_score = 0
else:
    pressure_score = st.slider("First Innings Target (to estimate pressure)", min_value=100, max_value=450, value=250)

user_input = pd.DataFrame([{
    'batsman': batsman,
    'batting_team': batting_team,
    'bowling_team': bowling_team,
    'venue': venue,
    'city': city,
    'season': season,
    'form_avg_3': form_avg_3,
    'form_avg_5': form_avg_5,
    'batting_position': batting_position,
    'innings': innings,
    'pressure_score': pressure_score
}])

# Replace categorical features to match updated form inputs
categorical_features = ['batsman', 'batting_team', 'bowling_team', 'venue', 'city']
feature_columns = metadata['feature_columns']
user_encoded = pd.get_dummies(user_input, columns=categorical_features)

for col in feature_columns:
    if col not in user_encoded.columns:
        user_encoded[col] = 0
user_encoded = user_encoded[feature_columns]

if st.button("Predict Runs"):
    predicted_runs = model.predict(user_encoded)[0]
    st.success(f"🎯 Predicted Runs: {round(predicted_runs, 2)}")
