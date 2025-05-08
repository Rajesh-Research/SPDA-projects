import streamlit as st
import pandas as pd

# Load dataset
file_path = "IPL_Histroy_2008-2024.csv"
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()
df["dismissal_kind"] = df["dismissal_kind"].str.strip()

# Get unique player names
player_list = sorted(set(df["batter"].dropna().unique()) | set(df["bowler"].dropna().unique()) | set(df["fielder"].dropna().unique()))

# Sidebar Navigation
st.sidebar.title("🏏 IPL Performance Dashboard")
menu = st.sidebar.radio("Go to", ["Home", "Season-wise Player Analysis", "Leaderboard", "Match-wise Performance"])

# --- Home ---
if menu == "Home":
    st.title("🏏 IPL Player Performance Dashboard")
    st.write("Welcome! Use the sidebar to navigate through different sections.")

# --- Season-wise Player Analysis ---
elif menu == "Season-wise Player Analysis":
    st.title("📊 Season-wise Player Analysis")
    player_name = st.selectbox("Select a player", player_list)

    if player_name:
        batting_data = df[df["batter"].str.contains(player_name, case=False, na=False)]
        total_runs = batting_data["batsman_runs"].sum()
        balls_faced = batting_data.shape[0]

        dismissals = df[df["player_dismissed"].str.contains(player_name, case=False, na=False)].shape[0]
        batting_average = total_runs / dismissals if dismissals > 0 else total_runs
        strike_rate = (total_runs / balls_faced) * 100 if balls_faced > 0 else 0

        runs_per_match = batting_data.groupby("match_id")["batsman_runs"].sum()
        fifties = runs_per_match[(runs_per_match >= 50) & (runs_per_match < 100)].count()
        hundreds = runs_per_match[runs_per_match >= 100].count()
        highest_score = runs_per_match.max()

        fours = batting_data[batting_data["batsman_runs"] == 4].shape[0]
        sixes = batting_data[batting_data["batsman_runs"] == 6].shape[0]
        total_matches = batting_data["match_id"].nunique()

        bowling_data = df[df["bowler"].str.contains(player_name, case=False, na=False)]
        wickets_data = df[(df["bowler"].str.contains(player_name, case=False, na=False)) & (df["is_wicket"] == 1) & (df["dismissal_kind"] != "run out")]

        total_wickets = wickets_data.shape[0]
        runs_conceded = bowling_data["total_runs"].sum()
        balls_bowled = bowling_data.shape[0]

        economy_rate = (runs_conceded / (balls_bowled / 6)) if balls_bowled > 0 else 0
        bowling_average = (runs_conceded / total_wickets) if total_wickets > 0 else 0
        bowling_strike_rate = (balls_bowled / total_wickets) if total_wickets > 0 else 0

        wickets_per_match = wickets_data.groupby("match_id").size()
        best_bowling_wickets = wickets_per_match.max() if not wickets_per_match.empty else 0

        fielder_data = df[df["fielder"].str.contains(player_name, case=False, na=False)]
        catches = fielder_data[fielder_data["dismissal_kind"] == "caught"].shape[0]
        stumpings = fielder_data[fielder_data["dismissal_kind"] == "stumped"].shape[0]
        run_outs = df[(df["dismissal_kind"] == "run out") & (df["fielder"].str.contains(player_name, case=False, na=False))].shape[0]
        total_dismissals = catches + stumpings + run_outs

        st.subheader("🏏 Batting Performance")
        st.table(pd.DataFrame({
            "Matches": [total_matches],
            "Runs": [total_runs],
            "Highest Score": [highest_score],
            "Average": [f"{batting_average:.2f}"],
            "Strike Rate": [f"{strike_rate:.2f}"],
            "50s": [fifties],
            "100s": [hundreds],
            "Fours": [fours],
            "Sixes": [sixes]
        }))

        st.subheader("🎯 Bowling Performance")
        st.table(pd.DataFrame({
            "Wickets": [total_wickets],
            "Best Bowling": [f"{best_bowling_wickets}"],
            "Average": [f"{bowling_average:.2f}" if total_wickets > 0 else "-"],
            "Economy": [f"{economy_rate:.2f}" if balls_bowled > 0 else "-"],
            "Strike Rate": [f"{bowling_strike_rate:.2f}" if total_wickets > 0 else "-"]
        }))

        st.subheader("🧤 Fielding Performance")
        st.table(pd.DataFrame({
            "Catches": [catches],
            "Stumpings": [stumpings],
            "Run Outs": [run_outs],
            "Total Dismissals": [total_dismissals]
        }))

# --- Leaderboard ---
elif menu == "Leaderboard":
    st.title("🏆 IPL Leaderboard")
    
    # Add "All Seasons" option
    season_options = ["All Seasons"] + sorted(df["match_year"].unique())
    season = st.selectbox("Select a season", season_options)

    # Filter data for selected season
    if season == "All Seasons":
        season_data = df
    else:
        season_data = df[df["match_year"] == season]

    # Find top performers
    top_scorer = season_data.groupby("batter")["batsman_runs"].sum().idxmax()
    top_scorer_runs = season_data.groupby("batter")["batsman_runs"].sum().max()
    top_wicket_taker = season_data.groupby("bowler")["is_wicket"].sum().idxmax()
    top_wickets = season_data.groupby("bowler")["is_wicket"].sum().max()

    # Find Best Catcher (Most Catches)
    if not season_data[season_data["dismissal_kind"] == "caught"].empty:
        top_catcher = season_data[season_data["dismissal_kind"] == "caught"].groupby("fielder").size().idxmax()
        top_catches = season_data[season_data["dismissal_kind"] == "caught"].groupby("fielder").size().max()
    else:
        top_catcher = "No Data"
        top_catches = 0

    st.write(f"🏆 **Orange Cap** (Most Runs): 🟠 {top_scorer} - {top_scorer_runs} Runs")
    st.write(f"🏆 **Purple Cap** (Most Wickets): 🟣 {top_wicket_taker} - {top_wickets} Wickets")
    st.write(f"🏆 **Best Catcher** (Most Catches): 🧤 {top_catcher} - {top_catches} Catches")

# --- Match-wise Performance ---
elif menu == "Match-wise Performance":
    st.title("📊 Match-wise Player Performance")
    match_id = st.selectbox("Select a match", sorted(df["match_id"].unique()))
    match_data = df[df["match_id"] == match_id]
    st.dataframe(match_data)
