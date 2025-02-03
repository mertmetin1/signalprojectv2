

import os
import math
import pandas as pd
import numpy as np

from config import processed_dir
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_prophet_model(ticker):
    processed_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
    if not os.path.exists(processed_file):
        return None, None, None, (None, None, None)
    
    df = pd.read_csv(processed_file, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    
    prophet_df = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    prophet_df.dropna(inplace=True)
    
    train_cutoff = pd.to_datetime("2023-01-01")
    train_data = prophet_df[prophet_df['ds'] < train_cutoff]
    test_data  = prophet_df[prophet_df['ds'] >= train_cutoff]
    
    model = Prophet(daily_seasonality=True)
    model.fit(train_data)
    
    future_periods = (test_data['ds'].max() - train_data['ds'].max()).days
    future = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)
    
    merged = pd.merge(test_data, forecast[['ds','yhat']], on='ds', how='left')
    y_true = merged['y']
    y_pred = merged['yhat']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    
    return model, forecast, merged, (mae, rmse, r2)
