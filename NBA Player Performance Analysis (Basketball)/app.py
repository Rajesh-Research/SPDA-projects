import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# Load saved artifacts
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_team = joblib.load('label_encoder_team.pkl')
label_encoder_pos = joblib.load('label_encoder_pos.pkl')
df = pd.read_csv('nba_preprocessed.csv')

# Streamlit app styling and layout
st.set_page_config(layout="wide", page_title="NBA Performance Predictor 2023-24")
st.title("NBA Player Performance Predictor 2023-24 Season")
st.write("Welcome! Predict a player's points per game (PPG) and explore their stats. Select a player or input custom stats below.")

# Sidebar for navigation and info
st.sidebar.header("Navigation")
st.sidebar.subheader("About This App")
st.sidebar.write("This app uses a Random Forest model (MAE: 0.99, R²: 0.97) to predict a player's PPG based on 2023-24 NBA stats. Explore player profiles or test your own stats!")

# Stat descriptions (simplified for users)
stat_descriptions = {
    "PPG": "Points Per Game: Average points scored per game.",
    "RPG": "Rebounds Per Game: Average rebounds grabbed per game.",
    "APG": "Assists Per Game: Average assists made per game.",
    "SPG": "Steals Per Game: Average steals per game.",
    "BPG": "Blocks Per Game: Average blocks per game.",
    "MPG": "Minutes Per Game: Average minutes played per game.",
    "USG%": "Usage Percentage: How often a player is involved in plays (higher = more usage).",
    "TS%": "True Shooting %: Overall shooting efficiency (includes 2s, 3s, and free throws).",
    "FTA": "Free Throw Attempts: Average free throw attempts per game."
}

# Player Selection or Manual Input Tab
tab1, tab2 = st.tabs(["Select Player", "Manual Input"])

with tab1:
    st.subheader("Select a Player")
    player_name = st.selectbox("Choose a player to analyze:", df['NAME'].unique(), key="player_select")

    # Filter data for selected player
    player_data = df[df['NAME'] == player_name].iloc[0]

    # Features used in the model
    numerical_features = ['AGE', 'GP', 'MPG', 'USG%', 'FTA', 'FT%', '2PA', '2P%', '3PA', '3P%', 
                         'eFG%', 'TS%', 'RPG', 'APG', 'SPG', 'BPG', 'TPG']
    categorical_features = ['TEAM_encoded', 'POS_encoded']
    all_features = numerical_features + categorical_features

    # Prepare input for prediction
    X_numerical = player_data[numerical_features].values.reshape(1, -1)
    X_categorical = player_data[categorical_features].values.reshape(1, -1)
    X_numerical_scaled = scaler.transform(X_numerical)
    X_player = np.hstack((X_numerical_scaled, X_categorical))
    predicted_ppg = rf_model.predict(X_player)[0]

    # Display player profile
    st.subheader(f"{player_name}'s Profile")
    col1, col2 = st.columns(2)
    with col1:
        try:
            team = label_encoder_team.inverse_transform([int(player_data['TEAM_encoded'])])[0]
        except ValueError:
            team = "Unknown"
        try:
            pos = label_encoder_pos.inverse_transform([int(player_data['POS_encoded'])])[0]
        except ValueError:
            pos = "Unknown"
        st.write(f"**Team**: {team}")
        st.write(f"**Position**: {pos}")
        st.write(f"**Games Played**: {player_data['GP']}")
        st.write(f"**Minutes/Game**: {player_data['MPG']:.1f} ({stat_descriptions['MPG']})")
    with col2:
        st.write(f"**Age**: {player_data['AGE']:.1f}")
        st.write(f"**Usage %**: {player_data['USG%']:.1f} ({stat_descriptions['USG%']})")
        st.write(f"**True Shooting %**: {player_data['TS%']:.3f} ({stat_descriptions['TS%']})")

    # Key stats and prediction
    st.subheader("Performance Stats")
    col3, col4 = st.columns(2)
    with col3:
        st.write(f"**Actual PPG**: {player_data['PPG']:.1f} ({stat_descriptions['PPG']})")
        st.write(f"**Predicted PPG**: {predicted_ppg:.1f}")
    with col4:
        st.write(f"**Rebounds/Game**: {player_data['RPG']:.1f} ({stat_descriptions['RPG']})")
        st.write(f"**Assists/Game**: {player_data['APG']:.1f} ({stat_descriptions['APG']})")
        st.write(f"**Steals/Game**: {player_data['SPG']:.1f} ({stat_descriptions['SPG']})")
        st.write(f"**Blocks/Game**: {player_data['BPG']:.1f} ({stat_descriptions['BPG']})")
        st.write(f"**Free Throw Attempts**: {player_data['FTA']:.1f} ({stat_descriptions['FTA']})")

    # Visualization
    st.subheader("Visual Breakdown")
    stats_to_show = ['PPG', 'RPG', 'APG', 'SPG', 'BPG']
    player_stats = player_data[stats_to_show].to_dict()
    fig = px.bar(x=list(player_stats.keys()), y=list(player_stats.values()), 
                 labels={'x': 'Stat', 'y': 'Value'}, 
                 title=f"{player_name}'s Key Stats", height=400)
    st.plotly_chart(fig)

with tab2:
    st.subheader("Enter Custom Player Stats")
    with st.form(key="custom_input"):
        col5, col6 = st.columns(2)
        with col5:
            age = st.number_input("Age", min_value=18.0, max_value=50.0, value=25.0, step=0.1)
            games_played = st.number_input("Games Played", min_value=0, max_value=82, value=50, step=1)
            mpg = st.number_input("Minutes/Game", min_value=0.0, max_value=48.0, value=30.0, step=0.1)
            usage_pct = st.number_input("Usage %", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
            fta = st.number_input("Free Throw Attempts", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        with col6:
            ft_pct = st.number_input("Free Throw %", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
            two_pa = st.number_input("2-Point Attempts", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
            two_p_pct = st.number_input("2-Point %", min_value=0.0, max_value=100.0, value=50.0, step=0.1)  # Fixed syntax
            three_pa = st.number_input("3-Point Attempts", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
            three_p_pct = st.number_input("3-Point %", min_value=0.0, max_value=100.0, value=35.0, step=0.1)  # Fixed syntax
        efg_pct = st.number_input("Effective FG %", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        ts_pct = st.number_input("True Shooting %", min_value=0.0, max_value=100.0, value=55.0, step=0.1)
        rpg = st.number_input("Rebounds/Game", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        apg = st.number_input("Assists/Game", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        spg = st.number_input("Steals/Game", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        bpg = st.number_input("Blocks/Game", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        tpg = st.number_input("Turnovers/Game", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
        team_encoded = st.selectbox("Team (Encoded)", options=range(len(label_encoder_team.classes_)), 
                                   format_func=lambda x: label_encoder_team.inverse_transform([x])[0])
        pos_encoded = st.selectbox("Position (Encoded)", options=range(len(label_encoder_pos.classes_)), 
                                  format_func=lambda x: label_encoder_pos.inverse_transform([x])[0])

        submit_button = st.form_submit_button(label="Predict PPG")

    if submit_button:
        # Create custom data array
        custom_data = np.array([[age, games_played, mpg, usage_pct, fta, ft_pct, two_pa, two_p_pct, 
                                three_pa, three_p_pct, efg_pct, ts_pct, rpg, apg, spg, bpg, tpg, 
                                team_encoded, pos_encoded]])
        
        # Scale numerical features
        custom_numerical = custom_data[:, :17]  # First 17 columns are numerical
        custom_categorical = custom_data[:, 17:].astype(int)  # Last 2 columns are categorical
        custom_numerical_scaled = scaler.transform(custom_numerical)
        X_custom = np.hstack((custom_numerical_scaled, custom_categorical))
        
        # Predict
        predicted_ppg_custom = rf_model.predict(X_custom)[0]
        st.subheader("Custom Prediction Results")
        st.write(f"**Predicted PPG**: {predicted_ppg_custom:.1f}")
        st.write("Note: Predictions are based on the trained model and may vary from real outcomes.")

# Footer
st.sidebar.header("Contact")
st.sidebar.write("Questions? Contact: example@xai.com")
st.sidebar.write(f"Last Updated: {pd.Timestamp.today().strftime('%Y-%m-%d')}")