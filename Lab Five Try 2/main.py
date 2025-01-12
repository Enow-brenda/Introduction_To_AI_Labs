# Load the dataset
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.src.layers import LSTM, Dropout, Dense
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('synthetic_traffic_flow_with_weather.csv')

# Display the first few rows of the dataset
print("Dataset:")
print(data.head())

# Check for missing data
print("Missing Data:", data.isnull().sum())

# Ensure the data is sorted chronologically
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.sort_values('timestamp', inplace=True)

data['timestamp_numeric'] = data['timestamp'].astype(np.int64) // 10**9
# Extract time-based features from the timestamp
data['hour'] = data['timestamp'].dt.hour

data['day_of_week'] = data['timestamp'].dt.dayofweek
data['month'] = data['timestamp'].dt.month


data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
data['is_rush_hour'] = data['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)  # checking if the hour falls in the range


# Normalizing the weather data
weather_features = ['temperature', 'humidity', 'rain']
scaler = MinMaxScaler()
data[weather_features] = scaler.fit_transform(data[weather_features])



# Lagging the features
data['traffic_flow_lag_hour'] = data['traffic_flow'].shift(10)  # previous hour traffic flow at that intersection
data['traffic_flow_lag_day'] = data['traffic_flow'].shift(240)  # previous day traffic flow at that same hour and intersection

# Handle missing values created by lag features
data.fillna(0, inplace=True)


# transforming traffic flow
traffic_features = ['traffic_flow', 'traffic_flow_lag_hour', 'traffic_flow_lag_day']
data[traffic_features] = scaler.fit_transform(data[traffic_features])


data = data.drop(columns='timestamp',axis=1)
scaled_data = scaler.fit_transform(data)
scaled_data= pd.DataFrame(scaled_data,columns=data.columns)

X = scaled_data.drop(columns=['traffic_flow'],axis=1)
y = scaled_data['traffic_flow']
print(f'Traffic flow: {y[:5]}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data to 3D for LSTM [samples, timesteps, features]
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))


# Define the LSTM model
model = Sequential()

# LSTM Layer 1
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# return_sequences=True to pass the sequence output to the next LSTM layer
model.add(Dropout(0.3))

# LSTM Layer 2
model.add(LSTM(units=64, return_sequences=True))
# return_sequences=True to pass the sequence output to the next LSTM layer
model.add(Dropout(0.3))

# LSTM Layer 3
model.add(LSTM(units=32, return_sequences=True))
# return_sequences=True to pass the sequence output to the next LSTM layer
model.add(Dropout(0.3))

# LSTM Layer 2
model.add(LSTM(units=20))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(1, activation='linear'))  # Regression task (traffic flow prediction)

# Compile the model
# Correct way to use Adam with learning rate
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)  # Use the Adam optimizer from tf.keras.optimizers
model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['accuracy','mae'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate the model
loss,accuracy,mae = model.evaluate(X_test, y_test)

# Extract data from the history object
epochs = range(1, len(history.history['loss']) + 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_mae = history.history['mae']
val_mae = history.history['val_mae']

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(epochs, train_mae, label='Training MAE', marker='o', color='blue')
plt.plot(epochs, val_mae, label='Validation MAE', marker='s', color='red')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE vs Epoch')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()

print(f'Model Accuracy: {accuracy}')
print(f'Model Loss: {loss}')

# Predict traffic flow
y_pred = model.predict(X_test)
print(f'Predictions: {y_pred[:5]}')

# Compute RMSE and MAE
rmse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")


'''
new_data = pd.read_csv('synthetic_weather_accident_holiday_data_next_10hours.csv')
# Convert timestamp to datetime and add numeric timestamp
new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
new_data['timestamp_numeric'] = new_data['timestamp'].astype(np.int64) // 10**9

# Extract time-based features from the timestamp
new_data['hour'] = new_data['timestamp'].dt.hour
new_data['day_of_week'] = new_data['timestamp'].dt.dayofweek

# Create weekend indicator: 1 for weekend (Saturday/Sunday), 0 for weekdays
new_data['is_weekend'] = new_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Create rush hour indicator: 1 for rush hour (7-9 AM or 4-6 PM), 0 otherwise
new_data['is_rush_hour'] = new_data['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)

# Normalize the weather data (temperature, humidity, and rain)
weather_features = ['temperature', 'humidity', 'rain']
scaler = MinMaxScaler()
new_data[weather_features] = scaler.fit_transform(new_data[weather_features])


# Handle missing values created by lag features (fill with the previous value or use a specified strategy)
new_data.fillna(method='bfill', inplace=True)  # Backfill missing data
new_data.drop(columns='timestamp',axis=1)


new_data_scaled = scaler.transform(new_data)  # Apply the same scaler
new_data_reshaped = new_data_scaled.reshape((1, 1, len(new_data.columns)))  # Reshape for the LSTM model

# Predict the traffic congestion level for the future time period
prediction = model.predict(new_data_reshaped)
print(f"Predicted Traffic Congestion: {prediction}")
'''




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
plt.title('Rainy vs Traffic Flow')
plt.xlabel('Rainy')
plt.ylabel('Traffic Flow')

plt.tight_layout()
plt.show()

