#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set up page
st.set_page_config(page_title="Tennis Fan Dashboard", layout="wide")
st.title("Understanding Fan Behavior—Tennis Fan Engagement & Business Metrics")

# Load Excel data
@st.cache_data
def load_data():
    df = pd.read_csv("tennis_fan_engagement_data.csv")
    return df

df = load_data()



# Sidebar filters
st.sidebar.header("Filter Options")
gender = st.sidebar.multiselect("Select Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
age = st.sidebar.slider("Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (20, 50))

filtered_df = df[(df["Gender"].isin(gender)) & (df["Age"] >= age[0]) & (df["Age"] <= age[1])]

# KPI Section
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Average Revenue", f"${filtered_df['Revenue_Contribution'].mean():.2f}")
col2.metric("Average Retention Score", f"{filtered_df['Retention_Score'].mean():.2f}")
col3.metric("Average Match Attendance", f"{filtered_df['Match_Attendance'].mean():.0f}")

# Distribution and Scatter Plot
st.markdown("---")
left, right = st.columns(2)

with left:
    st.markdown("Engagement Metric Distribution")
    metric = st.selectbox("Choose Metric", ["Match_Attendance", "Streaming_Hours", "App_Usage_Frequency", "Social_Media_Interactions"])
    fig1 = px.histogram(filtered_df, x=metric, nbins=20, color_discrete_sequence=["#00CC96"])
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.markdown("Revenue vs Retention Score")
    fig2 = px.scatter(filtered_df, x="Revenue_Contribution", y="Retention_Score", color="Gender", size="Merchandise_Spend")
    st.plotly_chart(fig2, use_container_width=True)

# Correlation Heatmap
st.markdown("---")
st.subheader("Correlation Heatmap")
fig3, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(filtered_df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig3)

# K-Means Clustering
st.markdown("---")
st.subheader("Fan Segmentation with K-Means Clustering")
features = st.multiselect("Select Features for Clustering", 
                          ["Match_Attendance", "Streaming_Hours", "App_Usage_Frequency", "Social_Media_Interactions", "Revenue_Contribution", "Merchandise_Spend"],
                          default=["Match_Attendance", "Revenue_Contribution"])
n_clusters = st.slider("Number of Clusters", 2, 6, 3)

if features:
    scaled_data = StandardScaler().fit_transform(filtered_df[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    filtered_df['Cluster'] = kmeans.fit_predict(scaled_data)

    fig4 = px.scatter(filtered_df, x=features[0], y=features[1], color="Cluster", symbol="Gender", title="Fan Clusters")
    st.plotly_chart(fig4, use_container_width=True)

    st.bar_chart(filtered_df["Cluster"].value_counts())

# Footer
st.markdown("---")
st.markdown("Interactive Dashboard — Tennis Fan Engagement")


# In[ ]:





# In[ ]:




