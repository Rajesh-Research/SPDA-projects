# app.py

import streamlit as st
import pandas as pd
import joblib

# === Load Model ===
MODEL_PATH = r'C:\Users\naveen\Downloads\IPL\ipl_auction_price_predictor_model.pkl'
model = joblib.load(MODEL_PATH)

# === Page Settings ===
st.set_page_config(page_title="IPL Auction Price Predictor", page_icon="🏏", layout="centered")
st.title("🏏 IPL Auction Price Predictor")
st.markdown("#### Enter player performance stats to predict their auction price.")

# === Input Form ===
with st.form("auction_form"):
    col1, col2 = st.columns(2)

    with col1:
        base_price = st.number_input("Base Price (in Cr)", min_value=0.0, max_value=10.0, value=1.0)
        matches = st.number_input("Matches Played", min_value=0, max_value=200, value=10)
        runs = st.number_input("Runs Scored", min_value=0, max_value=2000, value=100)
        wickets = st.number_input("Wickets Taken", min_value=0, max_value=100, value=10)
        fours = st.number_input("Fours", min_value=0, max_value=200, value=20)

    with col2:
        sixes = st.number_input("Sixes", min_value=0, max_value=150, value=5)
        batting_avg = st.number_input("Batting Average", min_value=0.0, max_value=100.0, value=30.0)
        batting_sr = st.number_input("Batting Strike Rate", min_value=0.0, max_value=250.0, value=130.0)
        bowling_avg = st.number_input("Bowling Average", min_value=0.0, max_value=100.0, value=30.0)
        economy_rate = st.number_input("Economy Rate", min_value=0.0, max_value=20.0, value=7.0)

    submit = st.form_submit_button("Predict Auction Price")

# === Prediction Logic ===
if submit:
    input_data = pd.DataFrame([{
        'Base Price': base_price,
        'MatchPlayed': matches,
        'RunsScored': runs,
        'Wickets': wickets,
        '4s': fours,
        '6s': sixes,
        'BattingAVG': batting_avg,
        'BattingS/R': batting_sr,
        'BowlingAVG': bowling_avg,
        'EconomyRate': economy_rate
    }])

    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Predicted Auction Price: ₹{predicted_price:,.2f} Cr")
