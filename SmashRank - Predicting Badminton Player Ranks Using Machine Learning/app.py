import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and dataset
rf_model = joblib.load("rank_prediction_model.pkl")
df = pd.read_csv(r"C:\Users\deepa\Downloads\SA Proj Fi\dataset\bwf_ranking_updated.csv")

st.title("🏸 Badminton Player Rank Predictor")

# User inputs
points = st.number_input("Enter total points:")
tournaments = st.number_input("Enter number of tournaments:")
confederation = st.selectbox("Select Confederation:", df["confederation"].unique())

if tournaments > 0 and points > 0:
    # Backend calculation
    points_per_tournament = points / tournaments
    input_data = np.array([[points_per_tournament]])

    # Predict World Rank
    predicted_rank = rf_model.predict(input_data)[0]

    # Calculate Confederation Rank
    new_row = pd.DataFrame([{
        'points_per_tournament': points_per_tournament,
        'confederation': confederation
    }])

    temp_df = pd.concat([df.copy(), new_row], ignore_index=True)

    temp_df['confed_rank'] = temp_df[temp_df['confederation'] == confederation]['points_per_tournament'] \
        .rank(ascending=False, method='min')
    predicted_confed_rank = temp_df.iloc[-1]['confed_rank']

    # Find Similar Players (within ±5%)
    similar_range = 0.05
    lower = points_per_tournament * (1 - similar_range)
    upper = points_per_tournament * (1 + similar_range)

    similar_players = df[(df['points_per_tournament'] >= lower) &
                         (df['points_per_tournament'] <= upper)]

    similar_players = similar_players[['rank', 'player', 'points', 'tournaments', 'confederation', 'country']]
    similar_players = similar_players.rename(columns={'rank': 'world_rank'})

    # Compute points_per_tournament for similar players
    similar_players['points_per_tournament'] = similar_players['points'] / similar_players['tournaments']

    # Display Results
    st.success(f"🌍 Predicted World Rank: {round(predicted_rank)}")
    st.info(f"🌐 Predicted {confederation} Rank: {int(predicted_confed_rank)}")

    st.subheader("🔍 Similar Players Based on Performance:")
    st.dataframe(similar_players.reset_index(drop=True))

    # Visualize user performance vs similar players
    st.subheader("📊 Player Performance vs Similar Players:")

    # Add the user data for comparison
    user_data = pd.DataFrame({
        'player': ['Your Player'],
        'points_per_tournament': [points_per_tournament],
        'confederation': [confederation]
    })

    # Combine user data with similar players for plotting
    combined_df = pd.concat([similar_players[['player', 'points_per_tournament']], user_data])

    # Plotting the comparison using a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='player', y='points_per_tournament', data=combined_df, palette='viridis')

    plt.title("Player Performance vs Similar Players")
    plt.xlabel("Player")
    plt.ylabel("Points per Tournament")
    plt.xticks(rotation=90)  # Rotate player names for better readability

    st.pyplot(plt)

else:
    st.warning("Please enter valid points and tournaments.")
