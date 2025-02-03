

import pandas as pd

def add_features(df):

    df['Pct_Change'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    
   
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
   
    df['Momentum'] = df['Close'].diff(10)
    
   
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    df.dropna(inplace=True)
    return df
