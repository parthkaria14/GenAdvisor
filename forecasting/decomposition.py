import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose

def stl_decompose(series, period):
    stl = STL(series, period=period)
    return stl.fit()

def classical_decompose(series, period):
    return seasonal_decompose(series, period=period)