import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Set Streamlit page config
st.set_page_config(page_title="Soccer Performance & Weather", layout="wide")

# Title
st.title("⚽ Weather Effects on Soccer Performance")
st.markdown("""
This interactive app predicts player performance based on weather and match context. 
Upload your dataset, explore the data, and get instant predictions and visualizations.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your player stats CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')
    df['kickoff_hour'] = df['kickoff_time'].dt.hour
    df['kickoff_day'] = df['kickoff_time'].dt.dayofweek
    df['is_weekend'] = df['kickoff_day'].isin([5, 6]).astype(int)
    df['cold_weather'] = (df['temperature_2m'] < 5).astype(int)
    df['hot_weather'] = (df['temperature_2m'] > 25).astype(int)
    df['wet_weather'] = (df['precipitation'] > 0.1).astype(int)
    df['good_performance'] = (df['total_points'] >= 5).astype(int)

    # Feature selection
    features = [
        'temperature_2m', 'precipitation', 'rain', 'snowfall',
        'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m',
        'kickoff_hour', 'kickoff_day', 'is_weekend',
        'hot_weather', 'cold_weather', 'wet_weather'
    ]

    df_model = df[features + ['good_performance']].dropna()
    X = df_model[features]
    y = df_model['good_performance']

    # Scale and split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Layout
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔍 Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
        st.markdown("### 📊 Feature Importances")
        importances = pd.Series(model.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        importances.plot(kind='barh', ax=ax, color='skyblue')
        st.pyplot(fig)

    with col2:
        st.markdown("### 📄 Classification Report")
        st.text(classification_report(y_test, y_pred))

    # Prediction tool
    st.markdown("---")
    st.markdown("## 🎯 Predict Performance for New Match")
    user_input = {}
    with st.form("prediction_form"):
        cols = st.columns(3)
        for i, feature in enumerate(features):
            default_val = float(np.round(df[feature].mean(), 2))
            user_input[feature] = cols[i % 3].number_input(feature.replace('_', ' ').title(), value=default_val)
        submitted = st.form_submit_button("Predict")

    if submitted:
        user_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(user_df)
        result = model.predict(user_scaled)[0]
        st.success("✅ Good Performance Expected") if result == 1 else st.warning("⚠️ Below-Average Performance Likely")

else:
    st.info("Please upload a CSV file to get started.")

