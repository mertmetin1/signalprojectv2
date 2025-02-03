

import os
import math
import pandas as pd
import numpy as np

from config import processed_dir
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_xgb_model(ticker):
    processed_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
    if not os.path.exists(processed_file):
        return None, (None, None, None), None, None
    
    df = pd.read_csv(processed_file, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    
    df['Target_Close'] = df['Close'].shift(-1)
    features = ['Close','Pct_Change','MA7','MA30','RSI','Momentum','Volatility']
    df.dropna(subset=features + ['Target_Close'], inplace=True)
    
    train_cutoff = pd.to_datetime("2023-01-01")
    train_data = df[df['Date'] < train_cutoff]
    test_data  = df[df['Date'] >= train_cutoff]
    
    X_train = train_data[features]
    y_train = train_data['Target_Close']
    X_test  = test_data[features]
    y_test  = test_data['Target_Close']
    
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    
    return model, (mae, rmse, r2), test_data, y_pred
