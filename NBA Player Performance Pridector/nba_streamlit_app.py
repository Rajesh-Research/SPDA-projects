
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.title("NBA Player Performance Predictor")
st.write("Upload a dataset and predict player's points per game (PTS) using a Random Forest Regressor.")

uploaded_file = st.file_uploader("Upload your NBA player stats CSV file", type=["csv"])

if uploaded_file:
    nba_data = pd.read_csv(uploaded_file)
    st.write("Initial data preview:")
    st.dataframe(nba_data.head())

    # Handle missing values
    nba_data.fillna({'FG%': 0, '3P%': 0, '2P%': 0, 'FT%': 0, 'eFG%': 0}, inplace=True)

    # Avoid data leakage by excluding target and related features
    features_to_exclude = ['PTS', 'FG', '3P']
    rolling_avg_columns = [col for col in nba_data.columns if 'Rolling' in col and 'PTS' in col]
    try:
        X = nba_data.drop(columns=features_to_exclude + rolling_avg_columns)
        y = nba_data['PTS']
    except KeyError:
        st.error("The dataset must contain a 'PTS' column for prediction.")
        st.stop()

    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    predictions = rf_model.predict(X_test_scaled)

    # Evaluate model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
