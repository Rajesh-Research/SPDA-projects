import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/Users/dileepkumar/Downloads/data.csv")
df.columns = df.columns.str.strip()

# Combine team1 and team2 into a single format
def restructure_data(df):
    team1 = df[['MATCHID', 'TEAM1', 'ATTACKS1', 'BLOCKS1', 'SERVES1',
                'OPPONENT_ERRORS1', 'TOTAL1', 'DIGS1', 'RECEPTION1', 'SETS1']]
    team1.columns = ['MATCHID', 'TEAM', 'ATTACKS', 'BLOCKS', 'SERVES',
                     'OPPONENT_ERRORS', 'TOTAL', 'DIGS', 'RECEPTION', 'SETS']
    
    team2 = df[['MATCHID', 'TEAM2', 'ATTACKS2', 'BLOCKS2', 'SERVES2',
                'OPPONENT_ERRORS2', 'TOTAL2', 'DIGS2', 'RECEPTION2', 'SETS2']]
    team2.columns = team1.columns

    return pd.concat([team1, team2], ignore_index=True)

# Metrics functions
def top_teams_by_total(df, top_n=10):
    return df.groupby('TEAM')['TOTAL'].sum().sort_values(ascending=False).head(top_n)

def top_teams_by_attacks(df, top_n=10):
    return df.groupby('TEAM')['ATTACKS'].sum().sort_values(ascending=False).head(top_n)

def efficiency_by_team(df, top_n=10):
    matches = df.groupby('TEAM')['MATCHID'].count()
    total = df.groupby('TEAM')['TOTAL'].sum()
    return (total / matches).sort_values(ascending=False).head(top_n)

# Streamlit App
def main():
    st.set_page_config(page_title="VNL Analytics", layout="wide")
    st.title("🏐 Volleyball Nations League - Team Performance Dashboard")

    # Data preparation
    df_teams = restructure_data(df)

    # Sidebar filter
    team_list = sorted(df_teams['TEAM'].unique())
    selected_team = st.sidebar.selectbox("Filter by Team", ["All"] + team_list)

    if selected_team != "All":
        df_teams = df_teams[df_teams["TEAM"] == selected_team]

    # Top teams by total
    st.subheader("🏆 Top Teams by Total Points")
    st.dataframe(top_teams_by_total(df_teams))

    # Top teams by attacks
    st.subheader("💥 Most Aggressive Teams (Attacks)")
    st.dataframe(top_teams_by_attacks(df_teams))

    # Efficiency (Total points per match)
    st.subheader("⚙️ Efficiency: Total Points per Match")
    st.dataframe(efficiency_by_team(df_teams))

    # Visualization
    st.subheader("📊 Top 10 Teams by Total Points (Bar Chart)")
    top_teams = top_teams_by_total(df_teams)
    fig, ax = plt.subplots()
    ax.bar(top_teams.index, top_teams.values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Total Points")
    st.pyplot(fig)

# Entry point
if __name__ == "__main__":
    main()
