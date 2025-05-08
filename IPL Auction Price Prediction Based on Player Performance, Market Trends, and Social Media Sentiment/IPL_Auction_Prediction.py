import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # Import joblib for saving the model

# === File paths ===
auction_data_path = r'C:\Users\naveen\Downloads\IPL\Auction Data\ipl_2022_dataset.csv'
performance_data_path = r'C:\Users\naveen\Downloads\IPL\Player Performance Dataset\IPL_Data.csv'
sentiment_data_path = r'C:\Users\naveen\Downloads\IPL\Twitter Sentiment Dataset\training.1600000.processed.noemoticon.csv'

# === Load datasets ===
auction_data = pd.read_csv(auction_data_path)
performance_data = pd.read_csv(performance_data_path)
sentiment_data = pd.read_csv(sentiment_data_path, encoding='ISO-8859-1')

print("Auction Data Columns:", auction_data.columns)
print("Performance Data Columns:", performance_data.columns)
print("Sentiment Data Sample:", sentiment_data.head(3))

# === Clean 'Base Price' column ===
def convert_base_price(price_str):
    if isinstance(price_str, str):
        price_str = price_str.strip()
        if 'Lakh' in price_str:
            return float(price_str.replace('Lakh', '').strip()) / 100
        elif 'Cr' in price_str:
            return float(price_str.replace('Cr', '').strip())
    return pd.to_numeric(price_str, errors='coerce')

auction_data['Base Price'] = auction_data['Base Price'].apply(convert_base_price)

# === Rename columns for merging ===
auction_data.rename(columns={'Player': 'Name'}, inplace=True)

# === Merge auction and performance data ===
merged = pd.merge(auction_data, performance_data, on='Name', how='inner')

# === Feature Selection (select useful numeric features only) ===
features = [
    'Base Price', 'MatchPlayed', 'RunsScored', 'Wickets',
    '4s', '6s', 'BattingAVG', 'BattingS/R', 'BowlingAVG',
    'EconomyRate'
]

target = 'COST IN ₹ (CR.)'

# === Drop rows with missing target or features ===
merged = merged[features + [target]].dropna()

# === Convert target to float ===
merged[target] = merged[target].replace('[^\d.]', '', regex=True).astype(float)

# === Train-test split ===
X = merged[features]
y = merged[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Predict ===
y_pred = model.predict(X_test)

# === Evaluate ===
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# === Save the model ===
joblib.dump(model, 'ipl_auction_price_predictor_model.pkl')
print("\nModel has been saved as 'ipl_auction_price_predictor_model.pkl'.")

