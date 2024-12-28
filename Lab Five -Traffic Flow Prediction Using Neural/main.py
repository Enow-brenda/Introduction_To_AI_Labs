# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.optimizer_v1 import Adam
import keras_tuner as kt

# Load the dataset
data = pd.read_csv('synthetic_traffic_data_yaounde.csv')

# Display the first few rows of the dataset
print("Dataset:")
print(data.head())

# Check for missing data
print("Missing Data:", data.isnull().sum())

# Handle missing data (example: fill missing values with the mean)
data.fillna(data.mean(), inplace=True)

# Ensure the data is sorted chronologically
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.sort_values('timestamp', inplace=True)

# Extract time-based features from the timestamp
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['month'] = data['timestamp'].dt.month
data['day_of_month'] = data['timestamp'].dt.day

# Display the first few rows of the preprocessed dataset
print("Preprocessed Data:", data.head())

data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
data['is_rush_hour'] = data['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)  # checking if the hour falls in the range

# Normalizing the weather data
weather_features = ['temperature', 'humidity', 'rain']
scaler = MinMaxScaler()
data[weather_features] = scaler.fit_transform(data[weather_features])

# Lagging the features
data['traffic_flow_lag1'] = data['traffic_flow'].shift(1)  # previous day traffic flow
data['traffic_flow_lag24'] = data['traffic_flow'].shift(24)

# Handle missing values created by lag features
data.fillna(0, inplace=True)

traffic_features = ['traffic_flow', 'traffic_flow_lag1', 'traffic_flow_lag24']
data[traffic_features] = scaler.fit_transform(data[traffic_features])

features = ['intersection_id', 'temperature', 'humidity', 'rain', 'hour', 'day_of_week', 'month', 'day_of_month', 'is_holiday', 'is_weekend', 'is_rush_hour', 'traffic_flow_lag1', 'traffic_flow_lag24']
target = 'traffic_flow'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Reshape the input data to 3D for LSTM [samples, timesteps, features]
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define the model building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='mean_squared_error')
    return model

# Initialize the Keras Tuner
tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=10, executions_per_trial=1, directory='my_dir', project_name='traffic_flow_tuning')

# Perform hyperparameter tuning
tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Display the summary of the best model
best_model.summary()

# Train the best model
history = best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Evaluate the best model using the testing data
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Test MSE: {mse}')
print(f'Test MAE: {mae}')
print(f'Test RMSE: {rmse}')

# Use the trained model to predict traffic congestion for future time periods
# Example: Predict traffic flow for the next 10 time periods
future_X = X_test[:10]  # Replace with actual future data
future_predictions = best_model.predict(future_X)

print("\nFuture Predictions:")
print(future_predictions)

# Step 7: Visualization

# 1. Visualize Traffic Predictions
# Plot the predicted vs. actual traffic flow
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual Traffic Flow')
plt.plot(y_pred, label='Predicted Traffic Flow')
plt.title('Actual vs Predicted Traffic Flow')
plt.xlabel('Time')
plt.ylabel('Traffic Flow')
plt.legend()
plt.show()

# 2. Weather vs. Traffic Flow Visualization
# Create scatter plots to visualize the relationship between weather conditions and traffic flow
plt.figure(figsize=(14, 7))

plt.subplot(3, 1, 1)
plt.scatter(data['temperature'], data['traffic_flow'], alpha=0.5)
plt.title('Temperature vs Traffic Flow')
plt.xlabel('Temperature')
plt.ylabel('Traffic Flow')

plt.subplot(3, 1, 2)
plt.scatter(data['humidity'], data['traffic_flow'], alpha=0.5)
plt.title('Humidity vs Traffic Flow')
plt.xlabel('Humidity')
plt.ylabel('Traffic Flow')

plt.subplot(3, 1, 3)
plt.scatter(data['rain'], data['traffic_flow'], alpha=0.5)
plt.title('Rain vs Traffic Flow')
plt.xlabel('Rain')
plt.ylabel('Traffic Flow')

plt.tight_layout()
plt.show()