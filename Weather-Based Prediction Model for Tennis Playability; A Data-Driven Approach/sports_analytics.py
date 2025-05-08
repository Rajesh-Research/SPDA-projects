import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# TRAINING DATA
# -----------------------------
play_data = pd.DataFrame({
    'outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Encode training data
le_dict = {}
for col in ['outlook', 'temp', 'humidity', 'wind', 'play']:
    le = LabelEncoder()
    play_data[col] = le.fit_transform(play_data[col])
    le_dict[col] = le

X = play_data[['outlook', 'temp', 'humidity', 'wind']]
y = play_data['play']

# Train a basic decision tree
model = DecisionTreeClassifier()
model.fit(X, y)

# -----------------------------
# Load MATCH DATA
# -----------------------------
match_data = pd.read_csv(r"C:\Users\saipr\OneDrive\Desktop\TERM-5\GIP 2ND YEAR\amul.doc\Data.csv", encoding='latin1')
locations = sorted(match_data['Location'].dropna().unique())

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Dynamic Tennis Match Predictor", page_icon="🎾")
st.title("🎾 Dynamic Match Feasibility Predictor")
st.markdown("Determine if a tennis match will be **played or postponed** based on **live weather** at any ATP match location.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🏟 Match Setup")
location = st.sidebar.selectbox("Match Location", locations)
date = st.sidebar.date_input("Match Date", datetime.date.today())
time = st.sidebar.time_input("Match Time", datetime.datetime.now().time())

# -----------------------------
# Weather API Integration
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🌦 Live Weather")

api_key = "62ccc8a97d8d25c1094dac65e7ac8c3f"
weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"

try:
    response = requests.get(weather_url)
    data = response.json()

    if response.status_code == 200:
        weather_condition = data['weather'][0]['main']
        temperature = data['main']['temp']
        humidity_val = data['main']['humidity']
        wind_speed = data['wind']['speed'] * 3.6  # m/s to km/h

        st.sidebar.write(f"**Outlook:** {weather_condition}")
        st.sidebar.write(f"**Temperature:** {temperature} °C")
        st.sidebar.write(f"**Humidity:** {humidity_val}%")
        st.sidebar.write(f"**Wind:** {wind_speed:.2f} km/h")
    else:
        st.sidebar.error("Could not retrieve weather data.")
        weather_condition, temperature, humidity_val, wind_speed = "Sunny", 28, 60, 10

except Exception as e:
    st.sidebar.error(f"API error: {e}")
    weather_condition, temperature, humidity_val, wind_speed = "Sunny", 28, 60, 10

# -----------------------------
# Map to Model Categories
# -----------------------------
def map_outlook(o):
    o = o.lower()
    if 'rain' in o:
        return 'Rain'
    elif 'cloud' in o or 'overcast' in o:
        return 'Overcast'
    else:
        return 'Sunny'

def map_temp(t):
    if t >= 30:
        return 'Hot'
    elif t >= 20:
        return 'Mild'
    else:
        return 'Cool'

def map_humidity(h):
    return 'High' if h >= 75 else 'Normal'

def map_wind(w):
    return 'Strong' if w >= 20 else 'Weak'

# Categorize values
outlook = map_outlook(weather_condition)
temp_cat = map_temp(temperature)
humidity_cat = map_humidity(humidity_val)
wind_cat = map_wind(wind_speed)

# Encode the prediction input
encoded_input = pd.DataFrame({
    'outlook': [le_dict['outlook'].transform([outlook])[0]],
    'temp': [le_dict['temp'].transform([temp_cat])[0]],
    'humidity': [le_dict['humidity'].transform([humidity_cat])[0]],
    'wind': [le_dict['wind'].transform([wind_cat])[0]]
})

# -----------------------------
# Predict Result
# -----------------------------
pred = model.predict(encoded_input)[0]
result = le_dict['play'].inverse_transform([pred])[0]

st.subheader("📊 Prediction Result")
if result == 'Yes':
    st.success("✅ The match is likely to be **PLAYED**.")
else:
    st.error("❌ The match is likely to be **POSTPONED**.")

st.markdown(f"""
**Match Location:** {location}  
**Date & Time:** {date} {time}  
**Conditions:** `{outlook}` | `{temp_cat}` | `{humidity_cat}` humidity | `{wind_cat}` wind  
""")

st.caption("⚡️ Live prediction powered by OpenWeatherMap + DecisionTree AI")
