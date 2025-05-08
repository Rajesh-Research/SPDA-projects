import pandas as pd

def load_data(file_path=r"C:\Users\hemak\Documents\Python Scripts\#.doc\Data.csv"):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Clean column names
    return df

def top_goal_scorers(df, top_n=10):
    return df.sort_values("Goals", ascending=False).head(top_n)

def best_xg_performance(df, top_n=10):
    df["Goal_vs_xG"] = df["Goals"] - df["xG"]
    return df.sort_values("Goal_vs_xG", ascending=False).head(top_n)

def efficiency_stats(df):
    df["Goals_per_Min"] = df["Goals"] / df["Mins"]
    return df.sort_values("Goals_per_Min", ascending=False).head(10)

def filter_data(df, year=None, league=None, club=None):
    if year:
        df = df[df["Year"] == year]
    if league:
        df = df[df["League"] == league]
    if club:
        df = df[df["Club"] == club]
    return df
