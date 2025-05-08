
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Title of the app
st.title("NHL Team Stats - Simple Analysis")

# Let user input a website URL (we'll use a dummy one for now)
st.write("Enter the URL where the NHL team stats can be found:")
url = st.text_input("Website URL", "https://www.nhl.com/stats/teams")

# Button to load data
if st.button("Load Data"):

    # This part should fetch real data from the NHL site
    # But here, we are using example data to make it easier
    data = {
        "Team": ["Maple Leafs", "Oilers", "Bruins"],
        "Wins": [45, 42, 48],
        "Losses": [25, 28, 20],
        "Points": [95, 90, 100]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Show the table
    st.subheader("Team Stats Table")
    st.dataframe(df)

    # Basic bar chart for wins and losses
    st.subheader("Team Wins vs Losses")

    fig, ax = plt.subplots()
    ax.bar(df["Team"], df["Wins"], color='green', label="Wins")
    ax.bar(df["Team"], df["Losses"], bottom=df["Wins"], color='red', label="Losses")
    ax.set_ylabel("Total Games")
    ax.set_title("Wins and Losses by Team")
    ax.legend()
    st.pyplot(fig)

    # Logistic Regression to guess good teams
    st.subheader("Basic Prediction Example")

    # Make a column that shows how often they win
    df["WinRate"] = df["Wins"] / (df["Wins"] + df["Losses"])

    # Create a target column — 1 means good team (over 90 points), 0 means not
    df["GoodTeam"] = (df["Points"] > 90).astype(int)

    # Create model and train it
    model = LogisticRegression()
    model.fit(df[["WinRate"]], df["GoodTeam"])

    # Make predictions
    predictions = model.predict(df[["WinRate"]])

    # Show accuracy and confusion matrix
    st.write("Accuracy of Prediction:", accuracy_score(df["GoodTeam"], predictions))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(df["GoodTeam"], predictions))

    # Show what model predicted
    df["Prediction"] = predictions
    st.subheader("Prediction Results")
    st.write(df[["Team", "Points", "WinRate", "GoodTeam", "Prediction"]])

# Footer
st.write("---")
st.write("This is a basic NHL stats viewer app created using Streamlit.")
