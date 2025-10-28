import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(series, n_lags=20, epochs=50, batch_size=32, val_split=0.2, scaler=None):
    s = series.dropna().astype(float)
    scaler = scaler or MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(s.values.reshape(-1,1))
    X, y = [], []
    for i in range(n_lags, len(scaled)):
        X.append(scaled[i-n_lags:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, n_lags, 1)
    y = np.array(y)
    model = build_lstm((n_lags,1))
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=val_split, callbacks=[es], verbose=0)
    return model, scaler

def predict_lstm(model, scaler, history, n_lags=20, steps=1):
    seq = scaler.transform(history.dropna().values.reshape(-1,1))[:,0]
    preds = []
    for _ in range(steps):
        x = seq[-n_lags:].reshape(1, n_lags, 1)
        p = model.predict(x, verbose=0)[0,0]
        preds.append(p)
        seq = np.append(seq, p)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    return preds