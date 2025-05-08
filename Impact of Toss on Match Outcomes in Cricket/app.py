# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and encoders
rf_model = joblib.load('random_forest_model.pkl')
le_toss_winner = joblib.load('le_toss_winner.pkl')
le_toss_choice = joblib.load('le_toss_choice.pkl')
le_team1 = joblib.load('le_team1.pkl')
le_team2 = joblib.load('le_team2.pkl')
le_venue = joblib.load('le_venue.pkl')

# Load preprocessed dataset (optional, used only if needed for other purposes)
df = pd.read_csv('preprocessed_odi_matches.csv')

# Get original team names and venues from encoders
team_options = sorted(le_team1.classes_)  # Same as le_team2.classes_ since teams overlap
venue_options = sorted(le_venue.classes_)
toss_choices = ['Bat', 'Bowl']  # Original labels for toss decision

# Streamlit app title and description
st.title("Cricket Match Outcome Predictor")
st.write("Predict the likelihood of the toss winner winning an ODI match based on historical data!")

# Input section
st.header("Enter Match Details")

# Team and toss inputs (using human-readable names)
toss_winner = st.selectbox("Toss Winner", team_options)
toss_choice = st.selectbox("Toss Decision", toss_choices)
team1 = st.selectbox("Team 1", team_options)
team2 = st.selectbox("Team 2", team_options, index=1 if len(team_options) > 1 else 0)  # Avoid same team by default
venue = st.selectbox("Match Venue", venue_options)

# Runs scored inputs
team1_runs = st.number_input("Team 1 Runs Scored", min_value=0, max_value=500, value=200)
team2_runs = st.number_input("Team 2 Runs Scored", min_value=0, max_value=500, value=180)

# Prediction button
if st.button("Predict Outcome"):
    # Encode inputs using the label encoders
    toss_winner_encoded = le_toss_winner.transform([toss_winner])[0]

    # Map toss_choice to the encoded value
    toss_choice_map = {'Bat': le_toss_choice.transform(['Bat'])[0] if 'Bat' in le_toss_choice.classes_ else 0,
                       'Bowl': le_toss_choice.transform(['Bowl'])[0] if 'Bowl' in le_toss_choice.classes_ else 1}
    toss_choice_encoded = toss_choice_map[toss_choice]

    team1_encoded = le_team1.transform([team1])[0]
    team2_encoded = le_team2.transform([team2])[0]
    venue_encoded = le_venue.transform([venue])[0]

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Toss Winner': [toss_winner_encoded],
        'Toss Winner Choice': [toss_choice_encoded],
        'Team1 Name': [team1_encoded],
        'Team2 Name': [team2_encoded],
        'Team1 Runs Scored': [team1_runs],
        'Team2 Runs Scored': [team2_runs],
        'Match Venue (Stadium)': [venue_encoded]
    })

    # Make prediction
    prediction_prob = rf_model.predict_proba(input_data)[0]
    win_prob = prediction_prob[1] * 100  # Probability of toss winner winning (class 1)

    # Display result
    st.subheader("Prediction Result")
    st.write(f"The probability that **{toss_winner}** (toss winner) wins the match is: **{win_prob:.2f}%**")

    # Simple interpretation
    if win_prob > 50:
        st.success("Based on historical data, the toss winner is likely to win this match!")
    else:
        st.warning("Based on historical data, the toss winner is less likely to win this match.")

# Footer
st.write("---")
st.write("Built with ❤️ using historical ODI match data (1971-2024).")