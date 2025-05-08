import pandas as pd
from sklearn.linear_model import LinearRegression
import os

def load_and_process_data(file_path="match_info_data.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    df = df.dropna(subset=["player_of_match"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    player_awards = (
        df.groupby("player_of_match")["date"]
        .agg(total_awards="count", last_award_date="max")
        .reset_index()
        .rename(columns={"player_of_match": "player_name"})
    )

    today = pd.to_datetime("today")
    player_awards["days_since_last_award"] = (today - player_awards["last_award_date"]).dt.days
    player_awards["form_score"] = player_awards["total_awards"] * 10 - player_awards["days_since_last_award"] * 0.1

    return player_awards

def predict_form_and_worth(player_awards_df):
    X = player_awards_df[["total_awards", "days_since_last_award"]]
    y = player_awards_df["form_score"]
    model = LinearRegression().fit(X, y)

    player_awards_df["predicted_form"] = model.predict(X)
    player_awards_df["predicted_worth"] = player_awards_df["predicted_form"] * 1_00_000

    return player_awards_df
