
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load dataset from uploaded file
@st.cache_data
def load_data():
    return pd.read_csv("ipl_matches.csv")

df = load_data()

# Preprocess and train model
@st.cache_resource
def train_model(df):
    def parse_score(score):
        try:
            runs, wickets = score.split('/')
            return int(runs), int(wickets)
        except:
            return np.nan, np.nan

    df = df[df['result'].notna() & (df['result'] != 'No result')].copy()
    df[['1st_runs', '1st_wickets']] = df['1st_inning_score'].apply(parse_score).apply(pd.Series)
    df[['2nd_runs', '2nd_wickets']] = df['2nd_inning_score'].apply(parse_score).apply(pd.Series)

    df['first_batting_team'] = np.where(df['decision'] == 'BAT FIRST', df['toss_won'],
                                        np.where(df['toss_won'] == df['home_team'], df['away_team'], df['home_team']))
    df['second_batting_team'] = np.where(df['first_batting_team'] == df['home_team'], df['away_team'], df['home_team'])

    df['first_innings_runs'] = np.where(df['first_batting_team'] == df['home_team'], df['home_runs'], df['away_runs'])
    df['second_innings_runs'] = np.where(df['second_batting_team'] == df['home_team'], df['home_runs'], df['away_runs'])
    df['second_innings_wickets'] = np.where(df['second_batting_team'] == df['home_team'], df['home_wickets'], df['away_wickets'])
    df['second_innings_overs'] = np.where(df['second_batting_team'] == df['home_team'], df['home_overs'], df['away_overs'])

    df['target'] = df['first_innings_runs'] + 1
    df['runs_left'] = df['target'] - df['second_innings_runs']
    df['wickets_left'] = 10 - df['second_innings_wickets']
    df['overs_left'] = 20 - df['second_innings_overs']
    df['required_run_rate'] = df['runs_left'] / df['overs_left'].replace(0, np.inf)
    df['required_run_rate'] = df['required_run_rate'].replace([np.inf, -np.inf], 0)
    df['venue_advantage'] = (df['second_batting_team'] == df['home_team']).astype(int)
    df['win'] = (df['winner'] == df['second_batting_team']).astype(int)

    features = ['runs_left', 'wickets_left', 'overs_left', 'required_run_rate', 'venue_advantage']
    target = 'win'
    data = df[features + [target]].dropna()

    model = LogisticRegression(max_iter=1000)
    model.fit(data[features], data[target])
    return model, df

model, df = train_model(df)

# UI
st.title("🏏 IPL Match Predictor - Second Innings Win Probability")

teams = sorted(df['home_team'].dropna().unique())
home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", [t for t in teams if t != home_team])
toss_winner = st.selectbox("Toss Winner", [home_team, away_team])
decision = st.radio("Toss Decision", ['BAT FIRST', 'FIELD FIRST'])

st.markdown("### 📊 Second Innings Match Info")
second_runs = st.number_input("Runs Scored So Far", min_value=0, value=50)
second_wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2)
overs_completed = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
first_innings_score = st.number_input("First Innings Runs", min_value=0, value=160)

# Infer batting order
first_batting_team = toss_winner if decision == 'BAT FIRST' else (away_team if toss_winner == home_team else home_team)
second_batting_team = away_team if first_batting_team == home_team else home_team

# Calculate features
target = first_innings_score + 1
runs_left = max(0, target - second_runs)
wickets_left = 10 - second_wickets
overs_left = max(0.1, 20 - overs_completed)
required_rr = runs_left / overs_left
venue_advantage = 1 if second_batting_team == home_team else 0

# Prediction
input_data = pd.DataFrame([[runs_left, wickets_left, overs_left, required_rr, venue_advantage]],
                          columns=['runs_left', 'wickets_left', 'overs_left', 'required_run_rate', 'venue_advantage'])

if st.button("Predict Win Probability"):
    prob = model.predict_proba(input_data)[0][1]
    st.success(f"🔮 Win Probability for **{second_batting_team}**: **{prob*100:.2f}%**")
