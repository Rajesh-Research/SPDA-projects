import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv(r"C:\Users\saipr\OneDrive\Desktop\TERM-5\GIP 2ND YEAR\cricket_data.csv")

st.title("🏏 Player Performance & Worth Prediction App")

# Player selection
players = df['Player_Name'].unique()
selected_player = st.selectbox("Select a Player", players)

# Filter data for player
player_df = df[df['Player_Name'] == selected_player].sort_values(by='Year')

# Display stats
st.subheader(f"Performance of {selected_player}")
st.write(player_df)

# Plot Batting Strike Rate over years
fig, ax = plt.subplots()
ax.plot(player_df['Year'], player_df['Batting_Strike_Rate'], marker='o', label='Strike Rate')
ax.set_title("Batting Strike Rate Over Years")
ax.set_ylabel("Strike Rate")
ax.set_xlabel("Year")
ax.legend()
st.pyplot(fig)

# Ensure Year is numeric
player_df['Year'] = pd.to_numeric(player_df['Year'], errors='coerce')

# Drop any NaNs just in case
player_df = player_df.dropna(subset=['Year', 'Batting_Strike_Rate'])

# Predict future Strike Rate using Linear Regression
X = player_df[['Year']]
y = player_df['Batting_Strike_Rate']
model = LinearRegression()
model.fit(X, y)

next_year = X['Year'].max() + 1
future_strike_rate = model.predict(np.array([[next_year]]))[0]

st.subheader("📈 Predicted Future Batting Strike Rate")
st.write(f"Estimated Strike Rate for {next_year}: **{future_strike_rate:.2f}**")

# Convert necessary columns to numeric
numeric_columns = ['Runs_Scored', 'Wickets_Taken', 'Catches_Taken', 'Balls_Faced', 'Balls_Bowled']
for col in numeric_columns:
    player_df[col] = pd.to_numeric(player_df[col], errors='coerce')

# Estimate player worth
st.subheader("💰 Estimated Player Worth")
worth = (
    player_df['Runs_Scored'].sum() * 1.5 +
    player_df['Wickets_Taken'].sum() * 25 +
    player_df['Catches_Taken'].sum() * 10
)
st.write(f"Estimated Worth Score: **{worth:.0f} points**")

# Batting Accuracy = Runs / Balls Faced
st.subheader("🎯 Batting Accuracy")
batting_accuracy = (
    player_df['Runs_Scored'].sum() / player_df['Balls_Faced'].replace(0, np.nan).sum()
)
st.write(f"Batting Accuracy: **{batting_accuracy:.2f} runs per ball**")

# Bowling Accuracy = Wickets / Balls Bowled
st.subheader("🎯 Bowling Accuracy")
total_wickets = player_df['Wickets_Taken'].sum()
total_balls_bowled = player_df['Balls_Bowled'].replace(0, np.nan).sum()
if total_balls_bowled > 0:
    bowling_accuracy = total_wickets / total_balls_bowled
    st.write(f"Bowling Accuracy: **{bowling_accuracy:.4f} wickets per ball**")
else:
    st.write("No bowling data available.")

st.caption("Developed with ❤️ for cricket data insights")
