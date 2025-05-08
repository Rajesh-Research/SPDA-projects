#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import plotly.express as px

# Load the enhanced dataset
data = pd.read_excel("fan_engagement_1000rows.xlsx")

# Streamlit page configuration
st.set_page_config(page_title="🏸 Fan Engagement Dashboard", layout="wide")

# Custom styles
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .reportview-container .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #4B0082;
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏸 Fan Engagement & Ad Interaction Dashboard")
st.markdown("Analyze real-time fan behavior, device preferences, and ad engagement during badminton broadcasts.")

# Filters
regions = st.multiselect("Select Regions", options=data['Region'].unique(), default=data['Region'].unique())
devices = st.multiselect("Select Device Types", options=data['Device_Type'].unique(), default=data['Device_Type'].unique())

filtered_data = data[(data['Region'].isin(regions)) & (data['Device_Type'].isin(devices))]

# Graph 1: Fan Engagement Over Time
st.subheader("📈 Fan Engagement Over Time")
fig1 = px.line(filtered_data, x='Timestamp', y='Fan_Engagement_Score', color='Region', markers=True)
st.plotly_chart(fig1, use_container_width=True)

# Graph 2: Ad Clicks vs Live Views by Event
st.subheader("📊 Ad Clicks vs Live Views per Event")
fig2 = px.bar(filtered_data, x='Event', y=['Ad_Clicks', 'Live_Views'], barmode='group')
st.plotly_chart(fig2, use_container_width=True)

# Graph 3: Watch Time by Device and Region
st.subheader("🌍 Watch Time by Device and Region")
fig3 = px.sunburst(filtered_data, path=['Region', 'Device_Type'], values='Watch_Time_Minutes')
st.plotly_chart(fig3, use_container_width=True)

# Graph 4: Social Media vs In-App Engagement
st.subheader("📱 Social Mentions vs In-App Interactions")
fig4 = px.scatter(filtered_data, x='In_App_Interactions', y='Social_Media_Mentions',
                  color='Event', size='Fan_Engagement_Score')
st.plotly_chart(fig4, use_container_width=True)

# Graph 5: Engagement by Hour
st.subheader("🕒 Engagement by Time of Day")
filtered_data['Hour'] = pd.to_datetime(filtered_data['Timestamp']).dt.hour
fig5 = px.box(filtered_data, x='Hour', y='Fan_Engagement_Score', points="all")
st.plotly_chart(fig5, use_container_width=True)

# Summary statistics
st.subheader("📌 Summary Statistics")
st.dataframe(filtered_data.describe(), use_container_width=True)

st.success("✅ Live dashboard loaded with filters and extended visualizations!")


# In[ ]:





# In[ ]:




