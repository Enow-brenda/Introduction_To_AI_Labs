import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


# Function to generate random weather data
def generate_weather_data():
    temperature = round(random.uniform(20, 35), 1)  # Temperature in Celsius
    humidity = random.randint(50, 100)  # Humidity in percentage
    rain = round(random.uniform(0, 10), 1)  # Rainfall in mm
    return temperature, humidity, rain


# Generate synthetic data
num_records = 100000
start_date = datetime(2023, 1, 1)
data = []

for i in range(num_records):
    timestamp = start_date + timedelta(hours=i)
    intersection_id = random.randint(1, 10)  # Random intersection ID
    traffic_flow = random.randint(50, 500)  # Random traffic flow
    temperature, humidity, rain = generate_weather_data()
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    month = timestamp.month
    day_of_month = timestamp.day
    is_holiday = 1 if timestamp.date() in [datetime(2023, 1, 1).date(), datetime(2023, 12, 25).date()] else 0

    data.append(
        [timestamp, intersection_id, traffic_flow, temperature, humidity, rain, hour, day_of_week, month, day_of_month,
         is_holiday])

# Create DataFrame
columns = ['timestamp', 'intersection_id', 'traffic_flow', 'temperature', 'humidity', 'rain', 'hour', 'day_of_week',
           'month', 'day_of_month', 'is_holiday']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('synthetic_traffic_data_yaounde.csv', index=False)

# Display the first few rows of the dataset
print(df.head())