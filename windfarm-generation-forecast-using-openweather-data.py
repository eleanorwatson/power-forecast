
#import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Config
API_KEY = ""
lat = 53.9833
lon = -3.2833
capacity_mw = 100
num_simulations = 100

# Obtaining Data from OpenWeatherMap API
# url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
# response = requests.get(url)
# data = response.json()


# with open('forecast_data.json', 'w') as json_file:
#     json.dump(data, json_file, indent=4)

# Load the JSON file
with open('forecast_data.json', 'r') as file:
    data = json.load(file)

# Extract the list of forecasts
forecast_list = data['list']

# Normalize the nested JSON into a flat DataFrame
df = pd.json_normalize(
    forecast_list,
    record_path=None,
    meta=['dt', 'dt_txt'],
    errors='ignore'
)

# Flatten nested fields like 'main', 'weather', 'wind', etc.
df_main = pd.json_normalize(forecast_list, record_path=None,meta=['main','wind'])

# Load the JSON file
with open('forecast_data.json', 'r') as file:
    data = json.load(file)

# Extract the list of forecasts
forecast_list = data['list']

# Normalize the nested JSON into a flat DataFrame
df = pd.json_normalize(
    forecast_list,
    record_path=None,
    meta=['dt', 'dt_txt'],
    errors='ignore'
)

df_main = pd.json_normalize(forecast_list, record_path=None,meta=['main','wind'])
weather_expanded = pd.json_normalize(forecast_list, 'weather', ['dt'])

# Merge the expanded weather data back into the main DataFrame
df_main = pd.merge(df_main, weather_expanded, on='dt', how='left')

df_main.drop(columns=['weather'], inplace=True)
# Rename the columns for clarity (optional)
df_main.rename(columns={
    'id': 'weather_id',
    'main': 'weather_main',
    'description': 'weather_description',
    'icon': 'weather_icon'
}, inplace=True)


print(df_main.columns)

