import yfinance as yf
from .preprocessing import handle_missing
from .decomposition import stl_decompose
from .smoothing import ses_forecast
from .arima_module import fit_arima
from .lstm_model import build_lstm
from .arima_lstm_combo import arima_lstm_combo
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

df = yf.download("AAPL", period="1y")
series = handle_missing(df['Close'].squeeze())
result = stl_decompose(series, period=7)
forecast = ses_forecast(series, steps=5)


print("STL Decomposition Trend:")
print(result.trend.tail())

print("SES Forecast:")
print(forecast)


plt.figure(figsize=(12,6))
plt.plot(series, label="Original Series")
plt.plot(result.trend, label="Trend (STL)")
plt.plot(result.seasonal, label="Seasonal (STL)")
plt.legend()
plt.title("STL Decomposition")
plt.show()

recent = series[-20:]
forecast_index = pd.date_range(start=series.index[-1], periods=len(forecast)+1, freq=series.index.freq or 'B')[1:]

plt.figure(figsize=(8,4))
plt.plot(recent.index, recent, label="Recent Data")
plt.plot(forecast_index, forecast, label="SES Forecast", marker='o')
plt.legend()
plt.title("Simple Exponential Smoothing Forecast")
plt.show()

plt.plot(series)
plt.title("Original Series")
plt.show()

# Step 2: Differencing
result = adfuller(series.dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])
for key, value in result[4].items():
    print(f"Critical Value ({key}): {value}")

if result[1] < 0.05:
    print("Series is stationary (reject H0)")
    d = 0
else:
    print("Series is NOT stationary (fail to reject H0)")
    d = 1
if d == 1:
    diff_series = series.diff().dropna()
else:
    diff_series = series.dropna()

# Step 3: Plot ACF and PACF
plot_acf(diff_series, lags=30)
plt.title("ACF of Differenced Series")
plt.show()

plot_pacf(diff_series, lags=30)
plt.title("PACF of Differenced Series")
plt.show()
combined_forecast, models = arima_lstm_combo(series, arima_order=(1,1,1), n_lags=10, lstm_epochs=30, forecast_horizon=5)

# Create forecast index
forecast_index = pd.date_range(start=series.index[-1], periods=len(combined_forecast)+1, freq=series.index.freq or 'B')[1:]

plt.figure(figsize=(10,5))
plt.plot(series[-50:], label="Recent Data")

# Combine last actual point with forecast for a smooth transition
last_actual = series[-1]
extended_forecast = [last_actual] + list(combined_forecast)
extended_index = [series.index[-1]] + list(forecast_index)

plt.plot(extended_index, extended_forecast, label="ARIMA+LSTM Forecast", marker='o', color='orange')
plt.legend()
plt.title("ARIMA + LSTM Combined Forecast (Continued)")
plt.show()