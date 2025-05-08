import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/dileepkumar/Downloads/archive/men_single.csv")
    return df

df = load_data()

st.title("🏸 Badminton Men's Singles Analytics Dashboard")

# Sidebar filters
st.sidebar.header("Filter Players")
selected_country = st.sidebar.selectbox("Select Country", options=sorted(df['country'].unique()))
country_df = df[df['country'] == selected_country]

st.sidebar.markdown(f"**Total Players from {selected_country}:** {len(country_df)}")
selected_player = st.sidebar.selectbox("Select Player", options=sorted(country_df['player_name'].unique()))
filtered_df = country_df[country_df['player_name'] == selected_player]

# General Stats
st.subheader("Selected Player Overview")
st.dataframe(filtered_df.reset_index(drop=True))

# Top 10 players by points
top_players = df[df['country'] == selected_country].sort_values(by="points", ascending=False).head(10)
st.subheader(f"Top 10 Players by Points from {selected_country}")
fig, ax = plt.subplots()
sns.barplot(data=top_players, x="points", y="jorsey_name", hue="country", ax=ax)
ax.set_title(f"Top 10 Players by Points from {selected_country}")
st.pyplot(fig)

# Country level stats
st.subheader("Country-Level Summary")
country_stats = df.groupby('country').agg({
    'player_name': 'count',
    'points': 'sum',
    'ranking': 'mean'
}).rename(columns={'player_name': 'Number of Players', 'points': 'Total Points', 'ranking': 'Avg. Ranking'}).sort_values(by='Total Points', ascending=False)
st.dataframe(country_stats)

# Efficiency metric
st.subheader("Efficiency: Points per Tournament")
df['efficiency'] = df['points'] / df['tournaments']
top_eff = df.sort_values(by="efficiency", ascending=False).head(10)
st.dataframe(top_eff[['player_name', 'efficiency', 'points', 'tournaments']])

# Player Worth Estimation (mock formula)
st.subheader("Estimated Player Worth")
df['estimated_worth'] = df['points'] * 15 + df['efficiency'] * 500
top_worth = df.sort_values(by="estimated_worth", ascending=False).head(10)
st.dataframe(top_worth[['player_name', 'estimated_worth']])

# Player trend prediction (Linear Regression based on tournaments)
st.subheader("📈 Predict Future Points Based on Tournaments")
player_data = df[df['player_name'] == selected_player]

# Simulate historical data (since we lack time-based data)
x = np.array(player_data['tournaments']).reshape(-1, 1)
y = np.array(player_data['points']).reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)
x_future = np.arange(player_data['tournaments'].values[0], player_data['tournaments'].values[0]+10).reshape(-1, 1)
y_future = model.predict(x_future)

fig2, ax2 = plt.subplots()
ax2.plot(x_future, y_future, label='Predicted Points', color='green')
ax2.scatter(x, y, color='blue', label='Current')
ax2.set_title(f"Points Projection for {selected_player}")
ax2.set_xlabel("Tournaments Played")
ax2.set_ylabel("Points")
ax2.legend()
st.pyplot(fig2)

st.caption("This dashboard is an analytical view of Men's Singles Badminton data, highlighting player trends, potential, and performance across countries.")
