import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# reference
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn import preprocessing
import keras
from sklearn.metrics import mean_absolute_error,mean_squared_error
import tensorflow as tf
df = pd.read_csv( "/Users/rajeshgolchha/Downloads/ipl.csv")


# Encode categorical variables
label_encoders = {}
categorical_columns = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train the model
features = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler', 'runs', 'overs', 'runs_last_5', 'wickets_last_5']
target = 'total'
X = df.drop(['date','total','mid','striker','non-striker'], axis =1)
y = df['total']

# Train test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = keras.Sequential([
    keras.layers.Input( shape=(X_train_scaled.shape[1],)),  # Input layer
    keras.layers.Dense(512, activation='relu'),  # Hidden layer with 512 units and ReLU activation
    keras.layers.Dense(216, activation='relu'),  # Hidden layer with 216 units and ReLU activation
    keras.layers.Dense(1, activation='linear')  # Output layer with linear activation for regression
])

# Compile the model with Huber loss
huber_loss = tf.keras.losses.Huber(delta=1.0)  # You can adjust the 'delta' parameter as needed
model.compile(optimizer='adam', loss=huber_loss)  # Use Huber loss for regression

model.fit(X_train_scaled,y_train,epochs=50, batch_size=64, validation_data=(X_test_scaled,y_test))
model_losses = pd.DataFrame(model.history.history)
model_losses.plot()
predictions = model.predict(X_test_scaled)

mean_absolute_error(y_test,predictions)

with open("cricket_model2.pkl", "wb") as file:
    pickle.dump(model, file)
with open("label_encoders2.pkl", "wb") as file:
    pickle.dump(label_encoders, file)
print("done")
