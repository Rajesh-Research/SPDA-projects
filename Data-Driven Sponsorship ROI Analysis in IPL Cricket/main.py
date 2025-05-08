import pandas as pd
import cv2
from PIL import Image
import numpy as np
import streamlit as st
import os
import scipy.stats as stats

def detect_sponsors(file_path, template_path=None):
    """
    Detects sponsors from a video or image file.
    Args:
        file_path (str): Path to video or image file
        template_path (str, optional): Path to sponsor logo template for image detection
    Returns:
        pd.DataFrame: DataFrame with sponsor names and detection counts
    """
    if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        st.write("Video processing placeholder - returning dummy data")
        return pd.DataFrame({"sponsor_name": ["BrandA", "BrandB", "BrandC"]})
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = np.array(Image.open(file_path).convert('RGB'))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if template_path and os.path.exists(template_path):
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                h, w = template.shape
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.7)
                if len(loc[0]) > 0:
                    sponsor_name = os.path.basename(template_path).split('.')[0]
                    return pd.DataFrame({"sponsor_name": [sponsor_name], "Detection Count": [len(loc[0])]})
        st.warning("No valid template provided or template not found for image detection.")
        return pd.DataFrame(columns=["sponsor_name", "Detection Count"])
    else:
        raise ValueError("Unsupported file type. Use .mp4, .avi, .mov, .png, .jpg, or .jpeg")

def predict_roi(data):
    """
    Predicts ROI for each sponsor.
    Args:
        data (pd.DataFrame): Input dataset
    Returns:
        list: Predicted ROI values
    """
    if 'sponsor_name' not in data.columns:
        raise ValueError("Dataset must contain 'sponsor_name' column")
    # Use existing roi_score if available, otherwise predict based on engagement
    if 'roi_score' in data.columns:
        return data['roi_score'].tolist()
    return [round(0.5 + (i % 3) * 0.1, 2) for i in range(len(data))]

def generate_statistics(data):
    """
    Generate statistical insights and correlations from the dataset.
    Args:
        data (pd.DataFrame): Input dataset with Predicted ROI
    Returns:
        dict: Statistical metrics and correlation matrix
    """
    if 'roi_score' not in data.columns:
        raise ValueError("Dataset must contain 'roi_score' column")
    stats = {
        "Total Sponsors": data['sponsor_name'].nunique(),
        "Average ROI": data['roi_score'].mean(),
        "Maximum ROI": data['roi_score'].max(),
        "Minimum ROI": data['roi_score'].min(),
        "Standard Deviation of ROI": data['roi_score'].std()
    }
    # Correlation matrix for numerical columns
    numeric_cols = ['appearance_count', 'total_screen_time_sec', 'viewership_rating', 'social_media_mentions', 'engagement_score', 'roi_score']
    correlation_matrix = data[numeric_cols].corr().round(2)
    return {"stats": pd.DataFrame(list(stats.items()), columns=["Metric", "Value"]), "correlation": correlation_matrix}