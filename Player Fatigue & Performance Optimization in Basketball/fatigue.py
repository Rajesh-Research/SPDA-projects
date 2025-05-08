import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, \
    mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

# -------------------- Load and Preprocess Data --------------------
df = pd.read_csv('player_data.csv')
df.fillna(0, inplace=True)
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# Label fatigue levels
df['Fatigue_Level'] = df['MIN'].apply(lambda x: 'High' if x >= 35 else 'Moderate' if x >= 25 else 'Low')

# Add DEF column if missing
if 'DEF' not in df.columns:
    df['DEF'] = np.random.randint(0, 10, size=len(df))

selected_cols = ['PLAYER_NAME', 'GAME_DATE', 'MIN', 'PTS', 'REB', 'AST', 'TO', 'DEF', 'Fatigue_Level']
df = df[selected_cols]

# -------------------- Streamlit UI Setup --------------------
st.set_page_config(page_title="Basketball Fatigue Tracker", layout="wide")
st.title("Player Fatigue & Performance Optimization in Basketball🏀")

# -------------------- Sidebar: Predict Current Fatigue --------------------
st.sidebar.title("🧪 Predict Current Fatigue")

player_name_input = st.sidebar.selectbox("Select Player", df['PLAYER_NAME'].unique())
min_input = st.sidebar.slider("Minutes Played", 0, 48, 30)
pts_input = st.sidebar.slider("Points", 0, 81, 10)
reb_input = st.sidebar.slider("Rebounds", 0, 40, 5)
ast_input = st.sidebar.slider("Assists", 0, 30, 3)
to_input = st.sidebar.slider("Turnovers", 0, 15, 2)
def_input = st.sidebar.slider("Defense", 0, 10, 5)

input_data = [[min_input, pts_input, reb_input, ast_input, to_input, def_input]]

# Filter and prepare player history
player_hist = df[df['PLAYER_NAME'] == player_name_input]
features = ['MIN', 'PTS', 'REB', 'AST', 'TO', 'DEF']
X = player_hist[features]
y = player_hist['Fatigue_Level']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train Model (XGBoost)
# Train Model (XGBoost with multiclass config)
model_current = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    objective='multi:softmax',
    num_class=len(np.unique(y_encoded))
)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
model_current.fit(X_train, y_train)


# Prediction
if st.sidebar.button("Predict Fatigue"):
    pred = model_current.predict(input_data)
    fatigue_pred = le.inverse_transform(pred)[0]
    st.sidebar.success(f"🔮 Predicted Fatigue Level: {fatigue_pred}")

    st.sidebar.markdown("### 🧾 Suggested Action")
    if fatigue_pred == "High":
        st.sidebar.error("🚨 Suggestion: Reduce playing time and schedule recovery.")
    elif fatigue_pred == "Moderate":
        st.sidebar.warning("⚠️ Suggestion: Monitor load, consider partial recovery.")
    else:
        st.sidebar.success("✅ Suggestion: Continue current training plan.")

# -------------------- Trends --------------------
st.subheader("📈 Performance & Fatigue Trends")
selected_player = st.selectbox("Select Player for Analysis", df['PLAYER_NAME'].unique())
player_df = df[df['PLAYER_NAME'] == selected_player].sort_values(by='GAME_DATE')

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(player_df['GAME_DATE'], player_df['MIN'], marker='o', label='Minutes')
ax.plot(player_df['GAME_DATE'], player_df['PTS'], marker='x', label='Points')
ax.set_title(f"Minutes vs Points for {selected_player}")
ax.set_xlabel("Game Date")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(10, 3))
sns.scatterplot(x='GAME_DATE', y='MIN', hue='Fatigue_Level',
                palette={'Low': 'green', 'Moderate': 'orange', 'High': 'red'},
                data=player_df, ax=ax2)
ax2.set_title("Fatigue Level Over Time")
st.pyplot(fig2)

st.download_button(
    label="📥 Download Player Trend Data",
    data=player_df.to_csv(index=False),
    file_name=f"{selected_player}_trend.csv",
    mime='text/csv'
)

# -------------------- Future Prediction (RandomForestRegressor) --------------------
st.subheader("📅 Predict Performance & Fatigue Over Next 5 Years")

player_df['Date_Ordinal'] = player_df['GAME_DATE'].map(datetime.datetime.toordinal)


def predict_future(df, col):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[['Date_Ordinal']], df[col])
    y_pred = model.predict(df[['Date_Ordinal']])

    mae = mean_absolute_error(df[col], y_pred)
    mse = mean_squared_error(df[col], y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(df[col], y_pred)

    future_dates = pd.date_range(start=df['GAME_DATE'].max() + pd.Timedelta(days=1), periods=5 * 82, freq='7D')
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_preds = model.predict(future_ordinals)

    return future_dates, future_preds, mae, mse, rmse, r2


future_dates, min_pred, mae_min, mse_min, rmse_min, r2_min = predict_future(player_df, 'MIN')
_, pts_pred, mae_pts, mse_pts, rmse_pts, r2_pts = predict_future(player_df, 'PTS')

future_fatigue = ['High' if m >= 35 else 'Moderate' if m >= 25 else 'Low' for m in min_pred]


def performance_zone(p): return 'Excellent' if p >= 30 else 'Average' if p >= 15 else 'Poor'


performance_zones = [performance_zone(p) for p in pts_pred]

# Line Chart: Future Predictions
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(future_dates, min_pred, label='Predicted Minutes')
ax3.plot(future_dates, pts_pred, label='Predicted Points')
ax3.set_title("Predicted Minutes and Points (Next 5 Years)")
ax3.set_xlabel("Year")  # Changed the label for X-axis
ax3.set_ylabel("Performance Value")  # Changed the label for Y-axis
ax3.legend()
st.pyplot(fig3)

# Fatigue Risk Scatter Plot
fig4, ax4 = plt.subplots(figsize=(10, 3))
sns.scatterplot(x=future_dates, y=min_pred, hue=future_fatigue,
                palette={'Low': 'green', 'Moderate': 'orange', 'High': 'red'}, ax=ax4)
ax4.set_title("Fatigue Risk Trend Over 5 Years")
ax4.set_xlabel("Year")  # Changed the label for X-axis
ax4.set_ylabel("Predicted Minutes")  # Kept Y-axis as predicted minutes
st.pyplot(fig4)

future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_MIN': min_pred,
    'Predicted_PTS': pts_pred,
    'Fatigue_Level': future_fatigue,
    'Performance_Zone': performance_zones
})
st.download_button(
    label="📥 Download Future Predictions",
    data=future_df.to_csv(index=False),
    file_name=f"{selected_player}_future_predictions.csv",
    mime='text/csv'
)

st.subheader("📊 Future Predictions - Model Evaluation")
st.markdown("**Minutes Prediction**")
st.text(f"MAE: {mae_min:.2f}, MSE: {mse_min:.2f}, RMSE: {rmse_min:.2f}, R² Score: {r2_min:.2f}")
st.markdown("**Points Prediction**")
st.text(f"MAE: {mae_pts:.2f}, MSE: {mse_pts:.2f}, RMSE: {rmse_pts:.2f}, R² Score: {r2_pts:.2f}")
