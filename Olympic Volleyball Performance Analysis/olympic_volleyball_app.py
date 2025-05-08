
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title
st.title("🏐 Olympic Volleyball Stats Viewer")

# File uploader for the dataset
file = st.file_uploader("Upload OlympicHistory.csv file", type=["csv"])

if file is not None:
    # Load the CSV into a DataFrame
    df = pd.read_csv(file)

    # Show first few rows
    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Filter for Volleyball data only
    volleyball_df = df[df['Sport'] == 'Volleyball']

    # Drop unnecessary columns
    volleyball_df = volleyball_df[['Year', 'Team', 'Event', 'Medal']]

    # Remove duplicates
    volleyball_df = volleyball_df.drop_duplicates()

    # Show clean volleyball data
    st.subheader("Clean Volleyball Data")
    st.write(volleyball_df.head())

    # Number of teams participating over the years
    participation = volleyball_df.groupby('Year')['Team'].nunique().reset_index()
    participation.columns = ['Year', 'Number_of_Teams']

    # Line plot for participation
    st.subheader("Number of Participating Teams Over the Years")
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=participation, x='Year', y='Number_of_Teams', marker='o', ax=ax1)
    ax1.set_title('Number of Participating Volleyball Teams')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Teams')
    ax1.grid(True)
    st.pyplot(fig1)

    # Top 10 medal winning teams
    st.subheader("Top 10 Medal Winning Teams")
    medal_count = volleyball_df.dropna().groupby('Team')['Medal'].count().sort_values(ascending=False).head(10)

    fig2, ax2 = plt.subplots()
    medal_count.plot(kind='bar', color='gold', ax=ax2)
    ax2.set_title("Top 10 Medal-Winning Volleyball Teams")
    ax2.set_ylabel("Medal Count")
    st.pyplot(fig2)

    # Show the table too
    st.write(medal_count.reset_index().rename(columns={"Medal": "Total Medals"}))

# Footer
st.write("---")
st.caption("Simple Olympic Volleyball analysis using Streamlit.")
