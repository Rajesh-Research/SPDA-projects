import streamlit as st
import pickle
import numpy as np
import os
st.write("Current Working Directory:", os.getcwd())
# Load the saved .pkl files
with open('driver_encoder.pkl', 'rb') as f:
    le_driver = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('points_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app title and description
st.title("F1 Driver Points Predictor")
st.write("Curious how many points your favorite F1 driver might score this season? Fill in their details and get a prediction!")

# Input section
st.header("Driver Details")

# Driver selection
driver = st.selectbox("Pick a Driver", le_driver.classes_, help="Choose an F1 driver from the list!")

# Core inputs
prev_season_points = st.slider(
    "Last Season's Points", 0, 600, 100,
    help="Points they scored last year. Top drivers like Verstappen can get 400+, midfielders around 50-150."
)
num_races = st.slider(
    "Races This Season", 15, 25, 22,
    help="Number of races this season. Recent F1 seasons have 22-24 races."
)
team_points = st.slider(
    "Team's Total Points", 0, 1000, 300,
    help="Points the driver's team might score. Top teams (e.g., Red Bull) can reach 600+, midfield teams 100-300."
)
avg_position = st.slider(
    "Average Finish Place", 1, 20, 10,
    help="Their typical finishing spot. 1 is first, 10 is midfield, 20 is near the back."
)
podiums = st.slider(
    "Podium Finishes", 0, 20, 2,
    help="Times they finish in the top 3. Champions might get 10+, midfielders 0-3."
)

# Optional advanced inputs
with st.expander("More Options (Optional)"):
    st.write("Tweak these for a more precise prediction!")
    avg_points_per_season = st.slider(
        "Career Average Points", 0.0, 300.0, 50.0, step=0.1,
        help="Average points per season over their career. Top drivers average 150-200."
    )
    points_std = st.slider(
        "Result Consistency", 0.0, 200.0, 50.0, step=0.1,
        help="How steady are their points year-to-year? Lower is more consistent (e.g., 20-50 for top drivers)."
    )
    years_since_first = st.slider(
        "Years Racing in F1", 0, 20, 5,
        help="Years since their F1 debut. Rookies are 0, veterans like Hamilton are 15+."
)

# Prediction button
if st.button("Get Prediction!"):
    # Prepare 9 features
    driver_encoded = le_driver.transform([driver])[0]
    features = np.array([[driver_encoded, prev_season_points, num_races, avg_points_per_season, 
                          team_points, points_std, years_since_first, avg_position, podiums]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    # Display result
    st.success(f"Predicted Points for {driver}: **{prediction:.0f}**")
    st.write(f"We think {driver} could score around {prediction:.0f} points this season!")

# Footer
st.write("---")
st.write("Created by Ritik | Powered by F1 Data & XGBoost")