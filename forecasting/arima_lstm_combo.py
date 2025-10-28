from forecasting.arima_module import fit_arima
from forecasting.lstm_model import train_lstm, predict_lstm
from forecasting.preprocessing import handle_missing
import pandas as pd
import logging # <-- 1. Import logging

# --- 2. Set up a logger for this file ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# --- End of logger setup ---


def arima_lstm_combo(series, arima_order=(1,1,1), n_lags=10, lstm_epochs=30, forecast_horizon=5):
    
    logger.info("Starting ARIMA-LSTM combo model...") # <-- 3. Log key steps
    s = handle_missing(series, method="ffill").dropna()
    s = pd.Series(s)
    
    # --- ARIMA Section ---
    logger.info(f"Fitting ARIMA model with order {arima_order}...")
    arima_fit = fit_arima(s, order=arima_order)
    arima_forecast = arima_fit.forecast(steps=forecast_horizon)
    logger.info(f"ARIMA Forecast: {arima_forecast.values}") # <-- 4. Log the result
    
    # --- Residuals Section ---
    fitted = pd.Series(arima_fit.fittedvalues)
    fitted = fitted.reindex(s.index)
    resid = s - fitted
    resid = resid.dropna()
    
    # --- Debugging logs (converted from print) ---
    logger.debug(f"Original series length: {len(s)}")
    logger.debug(f"ARIMA fitted values length: {len(fitted)}")
    logger.debug(f"Residuals length: {len(resid)}")
    if len(resid.tail(10)) > 0:
        logger.debug(f"Last 10 residuals:\n{resid.tail(10)}")
    
    if len(resid) < n_lags + 1:
        logger.error("Not enough residuals for LSTM training. Check ARIMA fit or input data.")
        raise ValueError("Not enough residuals for LSTM training.")
    
    # --- LSTM Section ---
    logger.info(f"Training LSTM model on residuals with n_lags={n_lags}...")
    lstm_model, scaler = train_lstm(resid, n_lags=n_lags, epochs=lstm_epochs)
    last_hist = resid.iloc[-n_lags:]
    
    logger.info(f"Predicting LSTM residuals for {forecast_horizon} steps...")
    resid_pred = predict_lstm(lstm_model, scaler, last_hist, n_lags=n_lags, steps=forecast_horizon)
    logger.info(f"LSTM Residuals Forecast: {resid_pred}") # <-- 5. Log the result
    
    # --- Combination Section ---
    combined = arima_forecast.values + resid_pred
    logger.info(f"Final Combined Forecast: {combined}") # <-- 6. Log the final prediction
    
    return combined, {"arima": arima_fit, "lstm": lstm_model}