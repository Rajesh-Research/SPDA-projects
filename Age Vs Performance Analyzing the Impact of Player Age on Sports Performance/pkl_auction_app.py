
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="PKL 2023 Auction Analysis", layout="wide")
st.title("Pro Kabaddi League 2023 Auction Analysis")

# Load the dataset
df = pd.read_csv("Pro Kabaddi Auction 2023.csv")

# Display dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Dataset Info
st.subheader("Dataset Info")
buffer = []
df.info(buf=buffer)
info_str = "\n".join(buffer)
st.text(info_str)

# Check for missing values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Top 10 most expensive players
st.subheader("Top 10 Most Expensive Players")
top_paid = df.sort_values(by="Price", ascending=False).head(10)
st.dataframe(top_paid[["Player", "Team", "Category", "Price"]])

# Plot: Top 10 Most Expensive Players
st.subheader("Visualization: Top 10 Most Expensive Players")
fig1, ax1 = plt.subplots(figsize=(10,6))
sns.barplot(data=top_paid, x="Price", y="Player", hue="Team", dodge=False, ax=ax1)
ax1.set_title("Top 10 Most Expensive Players")
ax1.set_xlabel("Price (INR)")
ax1.set_ylabel("Player")
st.pyplot(fig1)

# Plot: Total Spending by Team
st.subheader("Visualization: Total Spending by Team")
team_summary = df.groupby("Team")["Price"].sum().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(12,6))
sns.barplot(x=team_summary.values, y=team_summary.index, palette="coolwarm", ax=ax2)
ax2.set_title("Total Spending by Team")
ax2.set_xlabel("Total Price (INR)")
ax2.set_ylabel("Team")
st.pyplot(fig2)

# Plot: Distribution of Player Roles
st.subheader("Visualization: Distribution of Player Roles")
fig3, ax3 = plt.subplots(figsize=(8,6))
sns.countplot(data=df, y="Category", order=df["Category"].value_counts().index, palette="Set2", ax=ax3)
ax3.set_title("Distribution of Player Roles")
ax3.set_xlabel("Count")
ax3.set_ylabel("Category")
st.pyplot(fig3)
