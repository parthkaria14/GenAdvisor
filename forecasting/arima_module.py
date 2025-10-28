import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def simulate_ar(phi, n=1000):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t-1] + np.random.normal()
    return pd.Series(x)

def simulate_ma(theta, n=1000):
    e = np.random.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = e[t] + theta * e[t-1]
    return pd.Series(x)

def fit_arima(series, order=(1,0,1)):
    model = ARIMA(series, order=order)
    return model.fit()