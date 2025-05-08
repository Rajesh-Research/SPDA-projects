# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load saved artifacts
combined_df = pd.read_csv(r"C:\Users\shrey\OneDrive\Desktop\New\Streamlit App\combined_dataset.csv")
le_home = joblib.load(r"C:\Users\shrey\OneDrive\Desktop\New\Streamlit App\le_home.pkl")
le_away = joblib.load(r"C:\Users\shrey\OneDrive\Desktop\New\Streamlit App\le_away.pkl")
scaler = joblib.load(r"C:\Users\shrey\OneDrive\Desktop\New\Streamlit App\scaler.pkl")
rf_model = joblib.load(r"C:\Users\shrey\OneDrive\Desktop\New\Streamlit App\rf_model.pkl")

# Convert date column to datetime
combined_df['date'] = pd.to_datetime(combined_df['date'])

# Get unique teams for dropdowns
teams = sorted(set(combined_df['home_team'].unique()).union(combined_df['away_team'].unique()))

# Streamlit app layout
st.title("Footy Insights MVP")
st.write("Predict international football match outcomes and explore historical stats!")

# Sidebar for prediction inputs
st.sidebar.header("Match Prediction")
home_team = st.sidebar.selectbox("Home Team", teams)
away_team = st.sidebar.selectbox("Away Team", teams)
neutral = st.sidebar.checkbox("Neutral Venue", value=False)

# Prediction function
def predict_outcome(home_team, away_team, neutral):
    # Encode teams
    home_encoded = le_home.transform([home_team])[0]
    away_encoded = le_away.transform([away_team])[0]
    
    # Get team stats from combined_df (using most recent available data)
    home_stats = combined_df[combined_df['home_team'] == home_team].iloc[-1] if not combined_df[combined_df['home_team'] == home_team].empty else pd.Series(0, index=['home_win_rate', 'home_avg_goals_scored', 'home_avg_goals_conceded'])
    away_stats = combined_df[combined_df['away_team'] == away_team].iloc[-1] if not combined_df[combined_df['away_team'] == away_team].empty else pd.Series(0, index=['away_win_rate', 'away_avg_goals_scored', 'away_avg_goals_conceded'])
    
    # Prepare feature array
    features = np.array([[home_encoded, away_encoded,
                          home_stats.get('home_win_rate', 0), away_stats.get('away_win_rate', 0),
                          home_stats.get('home_avg_goals_scored', 0), away_stats.get('away_avg_goals_scored', 0),
                          home_stats.get('home_avg_goals_conceded', 0), away_stats.get('away_avg_goals_conceded', 0),
                          int(neutral)]])
    
    # Scale numerical features
    numerical_cols = ['home_win_rate', 'away_win_rate', 'home_avg_goals_scored', 'away_avg_goals_scored',
                      'home_avg_goals_conceded', 'away_avg_goals_conceded']
    features_scaled = features.copy()
    features_scaled[:, 2:8] = scaler.transform(features[:, 2:8])
    
    # Predict
    prediction = rf_model.predict(features_scaled)[0]
    probs = rf_model.predict_proba(features_scaled)[0]
    
    outcome_map = {0: "Draw", 1: "Home Win", 2: "Away Win"}
    return outcome_map[prediction], probs

# Display prediction
if st.sidebar.button("Predict"):
    outcome, probs = predict_outcome(home_team, away_team, neutral)
    st.subheader("Prediction")
    st.write(f"Predicted Outcome: **{outcome}**")
    st.write(f"Confidence: Draw: {probs[0]:.2%}, Home Win: {probs[1]:.2%}, Away Win: {probs[2]:.2%}")

# Historical stats section
st.subheader("Historical Stats")
selected_team = st.selectbox("Select Team for Stats", teams)

# Filter data for selected team
team_data = combined_df[(combined_df['home_team'] == selected_team) | (combined_df['away_team'] == selected_team)]

# Basic stats
wins = len(team_data[(team_data['home_team'] == selected_team) & (team_data['outcome'] == 1)]) + \
       len(team_data[(team_data['away_team'] == selected_team) & (team_data['outcome'] == 2)])
draws = len(team_data[team_data['outcome'] == 0])
losses = len(team_data[(team_data['home_team'] == selected_team) & (team_data['outcome'] == 2)]) + \
         len(team_data[(team_data['away_team'] == selected_team) & (team_data['outcome'] == 1)])
total_matches = wins + draws + losses

st.write(f"**{selected_team} Stats**")
st.write(f"Matches Played: {total_matches}")
st.write(f"Wins: {wins} ({wins/total_matches:.2%})")
st.write(f"Draws: {draws} ({draws/total_matches:.2%})")
st.write(f"Losses: {losses} ({losses/total_matches:.2%})")

# Plot win/loss/draw trend over time
trend_data = team_data.groupby(team_data['date'].dt.year)['outcome'].value_counts().unstack().fillna(0)
trend_data.columns = ['Draw', 'Home Win', 'Away Win']
fig = px.line(trend_data, title=f"{selected_team} Outcome Trend Over Time")
st.plotly_chart(fig)

# Footer
st.write("Note: Prediction accuracy is ~54%. Future updates will improve this!")