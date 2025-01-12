from datetime import datetime, timedelta
import time
import random

import numpy as np
import pandas as pd
import requests

intersections = {
    1: {"latitude": 3.8480, "longitude": 11.5021},  # Messa Intersection
    2: {"latitude": 3.8672, "longitude": 11.5186},  # Bastos Intersection
    3: {"latitude": 3.8578, "longitude": 11.5165},  # Nlongkak Intersection
    4: {"latitude": 3.8450, "longitude": 11.5076},  # Rond-Point De l'Unité (Unity Roundabout)
    5: {"latitude": 3.8867, "longitude": 11.5065},  # Etoudi Intersection
    6: {"latitude": 3.8694, "longitude": 11.5149},  # Warda Intersection
    7: {"latitude": 3.8344, "longitude": 11.5000},  # Central Market Intersection
    8: {"latitude": 3.8443, "longitude": 11.4933},  # Bessengue Intersection
    9: {"latitude": 3.8804, "longitude": 11.5103},  # Polytechnique Intersection
    10: {"latitude": 3.8906, "longitude": 11.5217},  # Yaoundé Airport Intersection
}
def generate_weather_data(timestamp):
    month = timestamp.month
    rain_prob = 0.1
    if month >= 3 and month <=10:
        rain_prob = 0.5
        if month>=7 and month<=9:
            rain_prob = 0.8
    temperature = round(random.uniform(20, 35), 1)  # Temperature in Celsius
    humidity = random.randint(50, 100)  # Humidity in percentage
    rain = np.random.choice([0, 1], p=[1 - rain_prob, rain_prob]) # Rainfall in mm
    return temperature, humidity, rain
def get_weather_data(timestamp,intersection):

    # Convert to Unix timestamp (seconds since 1970-01-01 00:00:00)
    unix_timestamp = int(time.mktime(timestamp.timetuple()))

    coordinates = intersections.get(intersection)
    API_KEY ='0b3c381c088ff97e8d855c9b8c089c04'
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={coordinates['latitude']}&lon={coordinates['longitude']}&dt={unix_timestamp}&appid={API_KEY}&units=metric"

    # Send the request
    response = requests.get(url)
    if response.status_code == 200:
        # Parse and return the weather data
        weather_data = response.json()
        return weather_data
    else:
        print(f"Error fetching weather data: {response.status_code}")
        return None

def generate_timestamp():
    start_date = datetime(2023, 1, 1, 0, 0)
    end_date = datetime(2023, 12, 31, 23, 59)
    timestamp_list = []

    # Start from the start date and generate hourly timestamps
    current_time = start_date
    while current_time <= end_date:
        timestamp_list.append(current_time)
        current_time += timedelta(hours=1)

    return timestamp_list

def generate_traffic_flow(timestamp,is_holiday,accident,event):
    base_traffic_flow = 100
    # Traffic flow adjustments
    if is_holiday:
        # Decrease traffic during peak hours on holidays
        if 7 <= timestamp.hour < 9 or 16 <= timestamp.hour < 18:
            base_traffic_flow *= random.uniform(0.3, 0.6)  # Reduce traffic during peak hours (rush hours)
        else:
            base_traffic_flow *= random.uniform(0.7, 0.9)  # Slightly reduced traffic during off-peak hours

    # Adjust traffic flow if there's an accident or event
    if accident:
        base_traffic_flow *= 0.5  # Reduce flow due to accidents
    if event:
        base_traffic_flow *= 1.5  # Surge in traffic due to events

    return int(base_traffic_flow)  # Return the adjusted traffic flow as an integer



holidays = [
    "2025-01-01",  # New Year's Day
    "2025-02-11",  # Youth Day
    "2025-05-01",  # Labour Day
    "2025-05-20",  # National Day
    "2025-08-15",  # Assumption Day
    "2025-11-01",  # All Saints' Day
    "2025-12-25",  # Christmas Day
]
accident_prob = 0.1  # 10% chance of an accident
event_prob = 0.2  # 20% chance of an event
timestamps = generate_timestamp()
data = []


for timestamp in timestamps:
    for intersection in range(1, len(intersections) + 1):
        intersection_id = intersection
        is_holiday = 1 if timestamp.date().isoformat() in holidays else 0
        accident = np.random.choice([0, 1], p=[1 - accident_prob, accident_prob])
        event = np.random.choice([0, 1], p=[1 - event_prob, event_prob])
        traffic_flow = generate_traffic_flow(timestamp, is_holiday, accident, event)
        temperature, humidity, rain = generate_weather_data(timestamp)
        '''
         weather_data = get_weather_data(timestamp,intersection)
        temperature = weather_data["main"].get("temp", None)
        humidity = weather_data["main"].get("humidity", None)
        rain = 0

        weather_conditions = weather_data.get("weather", [])
        for condition in weather_conditions:
            if "Rain" in condition["main"]:
                rain = 1
        '''



        data.append(
            [timestamp, intersection_id, traffic_flow,is_holiday,accident,event,humidity,rain,temperature])

# Create DataFrame
columns = ['timestamp', 'intersection_id', 'traffic_flow', 'is_holiday','accident','event','humidity','rain','temperature']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('synthetic_traffic_flow_with_weather.csv', index=False)