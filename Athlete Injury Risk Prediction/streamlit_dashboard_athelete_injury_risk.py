import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model
rf_model = joblib.load('rf_model.pkl')

# Feature column names used during model training
feature_columns = ['Age', 'Games_Played_Last_Month', 'Avg_Training_Hours_Per_Week',
                   'Fitness_Score', 'Avg_Recovery_Days', 'Injury_History_Count',
                   'Injury_Frequency', 'Training_Load', 'Recovery_to_Training_Ratio',
                   'Player_ID_Encoded']  # Note: Make sure to encode Player ID properly

# 🎯 Streamlit UI
st.set_page_config(page_title="Athlete Injury Risk Predictor", layout="centered")
st.title("🏥 Athlete Injury Risk Predictor")
st.markdown("Use the form below to input player details and predict injury risk for the next game.")

with st.form("player_form"):
    st.header("📋 Player Information")

    player_id = st.text_input("Player ID (e.g., P0001)", max_chars=10)
    age = st.slider("Age", 16, 45, step=1)
    games_played = st.slider("Games Played Last Month", 0, 15, step=1)
    avg_training_hours = st.slider("Average Training Hours/Week", 0, 40)
    fitness_score = st.slider("Fitness Score (0-100)", 0, 100)
    avg_recovery_days = st.slider("Avg Recovery Days", 0.0, 14.0, step=0.1)
    past_injury_count = st.slider("Past Injury Count", 0, 10)
    injury_frequency = st.slider("Injury Frequency (per year)", 0.0, 10.0, step=0.1)
    training_load = st.slider("Training Load Score (0-100)", 0.0, 100.0, step=0.1)
    recovery_ratio = st.slider("Recovery to Training Ratio", 0.0, 2.0, step=0.01)

    submitted = st.form_submit_button("Predict Injury Risk")

# Encode player ID to numeric value
def encode_player_id(player_id):
    try:
        return int(player_id.strip("P").lstrip("0"))
    except:
        return 0  # fallback if ID is not valid

if submitted:
    # Prepare the input in correct order
    encoded_id = encode_player_id(player_id)
    input_data = np.array([[age, games_played, avg_training_hours, fitness_score,
                            avg_recovery_days, past_injury_count, injury_frequency,
                            training_load, recovery_ratio, encoded_id]])
    
    # Make prediction
    prediction_prob = rf_model.predict_proba(input_data)[0][1]  # Probability of injury
    prediction_label = "High Injury Risk" if prediction_prob >= 0.3 else "Low Injury Risk"

    st.subheader("📊 Prediction Result")
    st.success(f"✅ {prediction_label}")
    st.progress(int(prediction_prob * 100))

    # Input Summary
    st.subheader("🔍 Input Summary")
    st.json({
        "Player ID": player_id,
        "Age": age,
        "Games Played Last Month": games_played,
        "Avg Training Hours/Week": avg_training_hours,
        "Fitness Score": fitness_score,
        "Avg Recovery Days": avg_recovery_days,
        "Past Injury Count": past_injury_count,
        "Injury Frequency": injury_frequency,
        "Training Load Score": training_load,
        "Recovery to Training Ratio": recovery_ratio
    })

# 🧪 Sanity Check (Debug Only)
st.markdown("### 🧪 Model Sanity Check")
test_input_1 = np.array([[24, 5, 24, 31, 9.2, 7, 5.3, 62.9, 1.47, 1]])  # Likely higher risk
test_input_2 = np.array([[20, 1, 10, 90, 2.0, 0, 0.1, 20.0, 0.8, 2]])   # Likely lower risk

def risk_label(prob):
    return "High Injury Risk" if prob >= 0.5 else "Low Injury Risk"

try:
    prob1 = rf_model.predict_proba(test_input_1)[0][1]
    prob2 = rf_model.predict_proba(test_input_2)[0][1]

    st.write("📌 Test Prediction 1 (Injury-prone player):", risk_label(prob1))
    st.write("📌 Test Prediction 2 (Healthy player):", risk_label(prob2))
except Exception as e:
    st.error(f"⚠️ Error during sanity check: {e}")
