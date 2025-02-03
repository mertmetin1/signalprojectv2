

import os
import math
import numpy as np
import pandas as pd

from config import processed_dir
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_lstm_sequences(array, sequence_length=10):
    X, y = [], []
    for i in range(len(array) - sequence_length):
        X.append(array[i:i+sequence_length, :-1])
        y.append(array[i+sequence_length, -1])
    return np.array(X), np.array(y)

def train_lstm_model(ticker, seq_length=10):
    processed_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
    if not os.path.exists(processed_file):
        return None, (None,None,None), None, None, None, None, None
    
    df = pd.read_csv(processed_file, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    
    df['Target_Close'] = df['Close'].shift(-1)
    features = ['Close','Pct_Change','MA7','MA30','RSI','Momentum','Volatility','Target_Close']
    df.dropna(subset=features, inplace=True)
    
    train_cutoff = pd.to_datetime("2023-01-01")
    train_data = df[df['Date'] < train_cutoff].copy()
    test_data  = df[df['Date'] >= train_cutoff].copy()
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[features])
    test_scaled  = scaler.transform(test_data[features])
    
    X_train, y_train = create_lstm_sequences(train_scaled, seq_length)
    X_test,  y_test  = create_lstm_sequences(test_scaled,  seq_length)
    
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_length, X_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    y_pred = model.predict(X_test)
    
   
    full_zero = np.zeros((len(y_pred), len(features)))
    full_zero[:, -1] = y_pred.reshape(-1)
    inv_pred = scaler.inverse_transform(full_zero)
    y_pred_inv = inv_pred[:, -1]
    
    full_zero_test = np.zeros((len(y_test), len(features)))
    full_zero_test[:, -1] = y_test
    inv_test = scaler.inverse_transform(full_zero_test)
    y_test_inv = inv_test[:, -1]
    
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2   = r2_score(y_test_inv, y_pred_inv)
    
    return model, (mae, rmse, r2), history, test_data, y_test_inv, y_pred_inv, seq_length
