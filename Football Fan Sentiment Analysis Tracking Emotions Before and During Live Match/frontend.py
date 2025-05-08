import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from backend import load_data, top_goal_scorers, best_xg_performance, efficiency_stats, filter_data

st.set_page_config(page_title="Football Analytics", layout="wide")
st.title("⚽ Player Performance Analytics ")

# Directly load from existing dataset
df = load_data(r"C:\Users\hemak\Documents\Python Scripts\#.doc\Data.csv")

# Sidebar filters
st.sidebar.title("Filters")
selected_year = st.sidebar.selectbox("Select Year", sorted(df["Year"].unique()), index=0)
selected_league = st.sidebar.selectbox("Select League", ["All"] + sorted(df["League"].unique().tolist()))
clubs = df["Club"].dropna().astype(str).unique().tolist()
selected_club = st.sidebar.selectbox("Select Club", ["All"] + sorted(clubs))


filtered_df = filter_data(
    df,
    year=selected_year,
    league=None if selected_league == "All" else selected_league,
    club=None if selected_club == "All" else selected_club
)

st.subheader("🎯 Top Goal Scorers")
st.dataframe(top_goal_scorers(filtered_df))

st.subheader("📊 Best Performers vs xG")
st.dataframe(best_xg_performance(filtered_df))

st.subheader("⚡ Most Efficient (Goals per Minute)")
st.dataframe(efficiency_stats(filtered_df))

st.subheader("⚖️ Goals vs Expected Goals Chart")
top_players = top_goal_scorers(filtered_df, 10)
fig, ax = plt.subplots()
ax.bar(top_players["Player Names"], top_players["Goals"], label="Goals")
ax.bar(top_players["Player Names"], top_players["xG"], label="xG", alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Goals / Expected Goals")
plt.legend()
st.pyplot(fig)
