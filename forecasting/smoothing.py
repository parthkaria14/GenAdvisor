import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt

def moving_average(series, window=5):
    return series.rolling(window).mean()

def weighted_moving_average(series, weights):
    return series.rolling(len(weights)).apply(lambda x: np.dot(x, weights)/np.sum(weights), raw=True)

def ses_forecast(series, steps=1):
    model = SimpleExpSmoothing(series).fit()
    return model.forecast(steps)

def holt_forecast(series, steps=1):
    model = Holt(series).fit()
    return model.forecast(steps)

def holt_winters_forecast(series, steps=1, seasonal_periods=12):
    model = ExponentialSmoothing(series, seasonal_periods=seasonal_periods, trend='add', seasonal='add').fit()
    return model.forecast(steps)