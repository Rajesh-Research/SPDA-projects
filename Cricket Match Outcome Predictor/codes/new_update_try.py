import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

# Load model and encoders
with open("cricket_model2.pkl", "rb") as file:
    model = pickle.load(file)

with open("label_encoders2.pkl", "rb") as file:
    label_encoders = pickle.load(file)


# --- App Title ---
st.title("🏏 Cricket Target Score Predictor")
st.markdown("Predict final target based on current match situation")

# --- Layout in Columns ---
col1, col2 = st.columns(2)

with col1:
    venue = st.selectbox("Select Venue", label_encoders['venue'].classes_)
    bat_team = st.selectbox("Select Batting Team", label_encoders['bat_team'].classes_)
    batsman = st.selectbox("Select Batsman", label_encoders['batsman'].classes_)
    runs = st.number_input("Current Runs", min_value=0)
    runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0, max_value=runs)

with col2:
    bowl_team = st.selectbox("Select Bowling Team", label_encoders['bowl_team'].classes_)
    bowler = st.selectbox("Select Bowler", label_encoders['bowler'].classes_)
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
    wickets = st.number_input("Wickets", min_value=0, max_value=10)
    wickets_last_5 = st.number_input("Wickets in Last 5 Overs", min_value=0, max_value=wickets)

# --- Validation ---
if bat_team == bowl_team:
    st.error("Batting and Bowling teams cannot be the same.")
elif batsman == bowler:
    st.error("Batsman and Bowler cannot be the same.")
else:
    # Encode input
    encoded_input = [
        label_encoders['venue'].transform([venue])[0],
        label_encoders['bat_team'].transform([bat_team])[0],
        label_encoders['bowl_team'].transform([bowl_team])[0],
        label_encoders['batsman'].transform([batsman])[0],
        label_encoders['bowler'].transform([bowler])[0],
        runs, wickets,overs, runs_last_5, wickets_last_5
    ]
    input=np.array(encoded_input)
    input = input.reshape(1,-1)
    # input = scaler.transform(input)
    if st.button("🎯 Predict Target Score"):
        prediction = model.predict(input)[0][0]
        if overs == 20.0 or wickets_last_5 == 10:
            st.success(f"Match over: Final Runs = {round(runs)}")
        else:
            st.success(f"Predicted Final Target Score: **{np.round(prediction)}**")
