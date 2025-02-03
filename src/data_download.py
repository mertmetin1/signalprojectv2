

import os
import yfinance as yf
import pandas as pd

from config import symbols, start_date, end_date, raw_dir

def download_data():
    print("Downloading DAta...")
    try:
        data = yf.download(symbols, start=start_date, end=end_date, group_by="ticker")
    except Exception as e:
        print("Error:", str(e))
        data = None

    if data is not None:
        for ticker in symbols:
            ticker_data = data[ticker].reset_index()
            ticker_data['Ticker'] = ticker
            file_path = os.path.join(raw_dir, f"{ticker}_data.csv")
            ticker_data.to_csv(file_path, index=False)
            print(f"[RAW] {ticker} -> {file_path} Saved.")
