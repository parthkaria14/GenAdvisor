import pandas as pd
import numpy as np
from scipy import stats

def handle_missing(series, method="ffill"):
    if method == "ffill":
        return series.fillna(method="ffill")
    elif method == "bfill":
        return series.fillna(method="bfill")
    elif method == "interpolate":
        return series.interpolate(method="linear")
    else:
        return series

def detect_outliers_zscore(series, threshold=3):
    z = np.abs(stats.zscore(series.dropna()))
    return series.index[z > threshold]

def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)].index