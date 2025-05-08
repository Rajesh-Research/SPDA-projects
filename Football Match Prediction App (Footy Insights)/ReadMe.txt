# Footy Insights MVP

This is a simple app to predict football match outcomes and view team stats. Follow these steps to run it on your Windows PC!

---

## What You Need
- Windows PC
- Python installed (download from [python.org](https://www.python.org/downloads/) if you don’t have it)

---

## Steps to Run

### 1. Unzip the File
- Find `Footy_Insights_App.zip` (e.g., in Downloads).
- Right-click it, choose **"Extract All"**, and pick a folder (e.g., Desktop).
- Open the extracted folder (e.g., `Footy_Insights_App`).

### 2. Open Command Prompt
- Press `Win + R`, type `cmd`, and hit Enter.
- In CMD, go to the folder: cd C:\Users\YourUsername\Desktop\Footy_Insights_App

(Replace `YourUsername` with your actual username.)

### 3. Install Required Tools
- Type this in CMD and press Enter: pip install streamlit pandas numpy scikit-learn==1.5.0 joblib plotly

- Wait for it to finish (takes a few minutes).

### 4. Start the App
- Run: streamlit run footy_insights.py

- Open your browser and go to `http://localhost:8501`.
- Use the app to predict matches or check stats!

### 5. Stop the App
- In CMD, press `Ctrl+C` to close it.

---

## Using the App
- **Predict**: Pick two teams in the sidebar and click "Predict".
- **Stats**: Choose a team to see wins, losses, and a graph.

Enjoy! Prediction accuracy is about 54%.