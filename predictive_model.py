import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Configuration ---
DATA_DIR = "data"
MODELS_DIR = "models"
# Updated list to match Indian assets from data ingestion
ASSETS = ["NSEI", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "GOLDBEES.NS"]
TARGET_COLUMN = "close"
HORIZON = 30  # Predict 30 days into the future

def create_features(df):
    """Creates time-series features from the dataframe."""
    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekday'] = df.index.weekday
    
    # Lag features
    for lag in [1, 5, 10, 30]:
        df[f'lag_{lag}'] = df[TARGET_COLUMN].shift(lag)
        
    # Rolling window features
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df[TARGET_COLUMN].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[TARGET_COLUMN].rolling(window=window).std()
        
    df.dropna(inplace=True)
    return df

def train_model_for_asset(asset):
    """Trains an XGBoost model for a single asset."""
    print(f"Training model for {asset}...")
    # Adjust for file naming convention
    file_path = os.path.join(DATA_DIR, f"{asset}.csv")
    if not os.path.exists(file_path):
        print(f"Data file not found for {asset}. Skipping.")
        return None

    # Read CSV with proper handling of the specific format
    # The CSV has: Price,close,high,low,open,volume in first row
    # Then ticker info, then Date header, then actual data
    try:
        # Skip the first 3 rows (headers) and use proper column names
        df = pd.read_csv(file_path, skiprows=3, names=['Date', 'close', 'high', 'low', 'open', 'volume'])
        
        # Remove any rows where Date is NaN or empty
        df = df.dropna(subset=['Date'])
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Sort by date to ensure proper time series order
        df = df.sort_index()
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}. Skipping.")
        return None

    # Ensure the index is a DatetimeIndex before proceeding
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"Failed to create DatetimeIndex for {file_path}. Skipping.")
        return None

    df = create_features(df)
    
    if df.empty:
        print(f"Not enough data for {asset} after feature creation. Skipping.")
        return None
        
    # Define features (X) and target (y)
    features = [col for col in df.columns if col != TARGET_COLUMN]
    X = df[features]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=10,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate model
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"RMSE for {asset}: {rmse:.4f}")

    return model

def run_training():
    """Runs the model training process for all assets."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    for asset in ASSETS:
        model = train_model_for_asset(asset)
        if model:
            model_path = os.path.join(MODELS_DIR, f"{asset}_model.joblib")
            joblib.dump(model, model_path)
            print(f"Saved model for {asset} to {model_path}")

if __name__ == "__main__":
    # To run this script: python predictive_model.py
    run_training()

