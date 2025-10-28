import pandas as pd
from arch import arch_model

def fit_garch(series, p=1, q=1):
    model = arch_model(series, vol='GARCH', p=p, q=q)
    return model.fit(disp='off')