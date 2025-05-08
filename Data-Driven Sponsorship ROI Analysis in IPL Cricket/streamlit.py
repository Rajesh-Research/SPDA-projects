import streamlit as st
import pandas as pd
import cv2
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from main import detect_sponsors, predict_roi, generate_statistics
import io

# Streamlit UI Design
st.set_page_config(page_title="IPL Sponsorship ROI Analysis", layout="wide")
st.title("🏏 IPL Sponsorship ROI Analysis Dashboard")
st.image('ipl_banner.jpg', use_container_width=True)
st.markdown("""
    Welcome to the IPL Sponsor Analysis Tool! Upload match videos, images, or CSV data to analyze sponsor visibility 
    and predict Return on Investment (ROI) with in-depth statistics and recommendations.
    """)

st.sidebar.image('ipl_logo.png', width=150)
st.sidebar.title('IPL Sponsor Analysis')
st.sidebar.markdown('Analyze sponsorship impact and ROI efficiently!')
st.sidebar.header("Upload Data")

# Upload options
video_file = st.sidebar.file_uploader("📹 Upload Match Video", type=["mp4", "avi", "mov"])
image_file = st.sidebar.file_uploader("📸 Upload Sponsor Image", type=["png", "jpg", "jpeg"])
data_file = st.sidebar.file_uploader("📄 Upload Sponsor Data (CSV)", type=["csv"])
template_file = st.sidebar.file_uploader("🌐 Upload Sponsor Logo Template", type=["png", "jpg", "jpeg"])

# Initialize session state
if 'sponsor_df' not in st.session_state:
    st.session_state['sponsor_df'] = pd.DataFrame()
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None

# Process uploads
if video_file:
    st.video(video_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name
    try:
        st.write("Processing Video... Please wait ⏳")
        st.session_state['sponsor_df'] = detect_sponsors(temp_video_path)
        st.write("### Detected Sponsors:")
        st.dataframe(st.session_state['sponsor_df'])
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        os.remove(temp_video_path)

elif image_file and template_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_template:
        temp_template.write(template_file.read())
        temp_template_path = temp_template.name
    try:
        st.write("Processing Image... Please wait ⏳")
        st.session_state['sponsor_df'] = detect_sponsors(image_file, temp_template_path)
        st.write("### Detected Sponsors:")
        st.dataframe(st.session_state['sponsor_df'])
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
    finally:
        os.remove(temp_template_path)
elif image_file and not template_file:
    st.warning("Please upload a sponsor logo template for image detection.")

if data_file:
    try:
        st.session_state['dataset'] = pd.read_csv(data_file)
        st.write("### Uploaded Sponsor Data Preview:")
        st.dataframe(st.session_state['dataset'].head())
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")

# Comprehensive Results
st.header("📋 Comprehensive Sponsorship Analysis Results")
if st.session_state['dataset'] is None:
    st.warning("Please upload a CSV dataset to enable ROI prediction and statistics.")
else:
    dataset = st.session_state['dataset']
    
    # ROI Prediction
    st.subheader("🔍 ROI Prediction")
    try:
        st.write("Predicting Sponsorship ROI... 🔍")
        predictions = predict_roi(dataset)
        dataset['Predicted ROI'] = predictions
        st.write("### Prediction Results:")
        st.dataframe(dataset)
    except Exception as e:
        st.error(f"Error predicting ROI: {str(e)}")

    # Enhanced Statistical Visualizations
    st.subheader("📊 Statistical Visualizations")
    try:
        # ROI Distribution
        fig1 = px.histogram(dataset, x='roi_score', nbins=20, title="ROI Distribution",
                           labels={'roi_score': "ROI Score"}, color_discrete_sequence=['#FF6B6B'])
        st.plotly_chart(fig1)

        # Box Plot for ROI
        fig2 = px.box(dataset, y='roi_score', title="ROI Box Plot",
                     labels={'roi_score': "ROI Score"}, color_discrete_sequence=['#4ECDC4'])
        st.plotly_chart(fig2)

        # Scatter Plot: Appearance Count vs. ROI
        fig3 = px.scatter(dataset, x='appearance_count', y='roi_score', title="Appearance Count vs. ROI",
                         labels={'appearance_count': "Appearance Count", 'roi_score': "ROI Score"},
                         color_discrete_sequence=['#45B7D1'])
        st.plotly_chart(fig3)

        # Sponsor-wise ROI Bar
        fig4 = px.bar(dataset, x='sponsor_name', y='roi_score', title="ROI by Sponsor",
                     labels={'sponsor_name': "Sponsor", 'roi_score': "ROI Score"},
                     color='roi_score', color_continuous_scale='Viridis')
        st.plotly_chart(fig4)

        # Pie Chart: Sponsor Type Distribution
        fig5 = px.pie(dataset, names='sponsor_type', title="Sponsor Type Distribution",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig5)

        # Heatmap: Correlation Matrix
        stats_result = generate_statistics(dataset)
        fig6 = go.Figure(data=go.Heatmap(
            z=stats_result['correlation'].values,
            x=stats_result['correlation'].columns,
            y=stats_result['correlation'].columns,
            colorscale='Viridis',
            zmin=-1, zmax=1
        ))
        fig6.update_layout(title="Correlation Heatmap of Sponsorship Metrics")
        st.plotly_chart(fig6)
    except Exception as e:
        st.error(f"Error generating visualizations: {str(e)}")

    # Statistical Insights
    st.subheader("📈 Statistical Insights")
    try:
        stats_result = generate_statistics(dataset)
        st.dataframe(stats_result['stats'])
        st.write("### Correlation Matrix:")
        st.dataframe(stats_result['correlation'])
    except Exception as e:
        st.error(f"Error generating statistics: {str(e)}")

    # Recommendations
    st.subheader("💡 Recommendations")
    avg_roi = dataset['roi_score'].mean()
    max_appearance = dataset['appearance_count'].max()
    mean_appearance = dataset['appearance_count'].mean()
    recommendations = []
    if max_appearance > mean_appearance * 1.5:
        recommendations.append("Focus on optimizing high-appearance sponsors for better ROI.")
    if avg_roi < 5.0:  # Adjusted threshold based on dataset ROI range (1-10)
        recommendations.append("Increase engagement or screen time to boost ROI above 5.0.")
    if recommendations:
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.write("No specific recommendations at this time. ROI is performing well!")

    # Integrate Sponsor Detection
    if not st.session_state['sponsor_df'].empty:
        st.subheader("🏆 Sponsor Detection Results")
        st.write("Detected sponsors from uploaded video or image:")
        st.dataframe(st.session_state['sponsor_df'])

st.markdown("---")
st.write("**Tip:** Use a CSV with 'sponsor_name', 'appearance_count', 'total_screen_time_sec', 'viewership_rating', 'social_media_mentions', 'engagement_score', and 'roi_score' for best results!")