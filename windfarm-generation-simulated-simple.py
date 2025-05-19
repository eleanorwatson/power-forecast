import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
capacity_mw = 100
hours = 24 * 7  # One week
np.random.seed(42)

# Simulate wind speed (m/s) forecast hourly over the next week.
# Assumption: a daily pattern with some randomness
base_speed = 8  # mean wind speed in m/s
wind_speeds = base_speed + 3 * np.sin(np.linspace(0, 2 * np.pi, hours)) + np.random.normal(0, 1.5, hours)
wind_speeds = np.clip(wind_speeds, 0, 30)


def wind_to_power(windspeed, capacity_mw):
    """
    converts wind speeds to power generated.
    TODO: Capacity mw has not been defined?
    """
    if windspeed < 3:
        return 0
    elif 3 <= windspeed < 12:
        return capacity_mw * ((windspeed - 3) / (12 - 3))  # linear ramp-up
    elif 12 <= windspeed <= 25:
        return capacity_mw
    else:
        return 0  # turbine shuts down

def power_to_energy(power_mw, granularity=1):
    """
    power in mw
    granularity in hours
    """
    energy_mwh = power_mw * granularity
    return energy_mwh

wind_to_power_vec = np.vectorize(wind_to_power)
power_output_mw = wind_to_power_vec(wind_speeds, capacity_mw)

energy_mwh = power_to_energy(power_output_mw)

# Plotting
dates = pd.date_range(start=pd.Timestamp.today().normalize(), periods=hours, freq='H')
df = pd.DataFrame({'Wind Speed (m/s)': wind_speeds, 'Power Output (MW)': power_output_mw}, index=dates)

# Aggregate daily for plot
daily_energy = df['Power Output (MW)'].resample('D').sum()  # MWh per day

# Plotting
plt.figure(figsize=(12, 6))
daily_energy.plot(kind='bar', color='skyblue')
plt.title('Predicted Daily Energy Production for Wind Farm (MWh)')
plt.ylabel('Energy (MWh)')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
