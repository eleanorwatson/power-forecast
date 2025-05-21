
#import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor


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

df = pd.json_normalize(
    forecast_list,
    record_path=None,
    meta=['dt', 'dt_txt'],
    errors='ignore'
)

df_main = pd.json_normalize(forecast_list, record_path=None,meta=['main','wind', 'weather'])

# Rename the columns for clarity (optional)
df_main.rename(columns={
    'id': 'weather_id',
    'main': 'weather_main',
    'description': 'weather_description',
    'icon': 'weather_icon'
}, inplace=True)


print(df_main.columns)

#Assumptions: 
# Windpower is proportional to the cube of windspeed scaled to the farms capacity. 

df_main['wind_power_mw'] = (df_main['wind.speed'] ** 3) / (df_main['wind.speed'].max() ** 3) * 100

# Features and target
features = ['main.temp', 'main.feels_like', 'main.temp_min', 'main.temp_max', 'main.pressure', 'main.sea_level', 'main.grnd_level',
            'main.humidity', 'wind.speed', 'wind.deg', 'wind.gust', 'visibility', 'pop']
target = 'wind_power_mw'

df_main = df_main.dropna(subset=features + [target])


X = df_main[features]
y = df_main[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and fit model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

#  Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE on test set: {rmse:.2f} MW")

# This is assumed forecast for next week. 
next_week_forecast = pd.DataFrame({
    'main.temp': [15, 14, 13, 12, 11, 10, 9],
    'main.feels_like': [14, 13, 12, 11, 10, 9, 8],
    'main.temp_min': [13, 12, 11, 10, 9, 8, 7],
    'main.temp_max': [16, 15, 14, 13, 12, 11, 10],
    'main.pressure': [1015, 1016, 1017, 1018, 1019, 1020, 1021],
    'main.sea_level': [1015, 1016, 1017, 1018, 1019, 1020, 1021],
    'main.grnd_level': [1014, 1015, 1016, 1017, 1018, 1019, 1020],
    'main.humidity': [80, 75, 70, 65, 60, 55, 50],
    'wind.speed': [5, 6, 7, 8, 9, 10, 11],
    'wind.deg': [200, 210, 220, 230, 240, 250, 260],
    'wind.gust': [7, 8, 9, 10, 11, 12, 13],
    'visibility': [10000, 10000, 10000, 10000, 10000, 10000, 10000],
    'pop': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
})

next_week_forecast['time'] = pd.date_range(start='2025-05-27', periods=7, freq='D')


next_week_forecast['predicted_wind_power_mw'] = model.predict(next_week_forecast[features])

print(next_week_forecast[['wind.speed', 'predicted_wind_power_mw']])

plt.figure(figsize=(10, 6))
plt.plot(next_week_forecast['time'], next_week_forecast['predicted_wind_power_mw'], marker='o', label='Predicted Wind Power (MW)')

plt.title('Wind Farm Generation Forecast (7 Days)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Predicted Wind Power (MW)', fontsize=12)
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()