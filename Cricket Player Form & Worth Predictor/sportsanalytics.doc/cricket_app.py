import streamlit as st
import matplotlib.pyplot as plt
from backend import load_and_process_data, predict_form_and_worth

# Streamlit settings
st.set_page_config(page_title="Cricket Player Analytics", layout="wide")
st.title("🏏 Cricket Player Form & Worth Predictor")

# Load data
try:
    df = load_and_process_data("match_info_data.csv")
    df = predict_form_and_worth(df)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Select player
player = st.selectbox("Select Player", df["player_name"].sort_values())
pdata = df[df["player_name"] == player].iloc[0]

# Layout: stats + prediction
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Player Stats")
    st.metric("Total Awards", pdata["total_awards"])
    st.metric("Days Since Last Award", int(pdata["days_since_last_award"]))
    st.metric("Form Score", round(pdata["form_score"], 2))

with col2:
    st.subheader("🔮 Prediction")
    st.metric("Predicted Form", round(pdata["predicted_form"], 2))
    st.metric("Estimated Worth", f"₹ {int(pdata['predicted_worth']):,}")

# Bar chart for top players
st.subheader("📈 Top Players by Predicted Form")
top_players = df.sort_values("predicted_form", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(top_players["player_name"], top_players["predicted_form"], color="darkgreen")
ax.set_xlabel("Predicted Form")
ax.set_title("Top 10 Players")
ax.invert_yaxis()
st.pyplot(fig)

st.caption("Form score is based on Player of the Match frequency and recency.")
