

import os
import pandas as pd


symbols = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'TSLA', '^GSPC', 'NQ=F', 'RTY=F', '^DJI']
start_date = "2000-01-01"
end_date   = "2025-01-01"

train_end   = pd.to_datetime("2020-01-01")
test_end    = pd.to_datetime("2023-01-01")
valid_end   = pd.to_datetime("2024-01-01")
pred_start  = pd.to_datetime("2024-01-01")
pred_end    = pd.to_datetime("2024-12-31")


data_dir        = "data"
raw_dir         = os.path.join(data_dir, "raw")
processed_dir   = os.path.join(data_dir, "processed")
predictions_dir = os.path.join(data_dir, "predictions")
reports_dir     = os.path.join(data_dir, "reports")


RANDOM_SEED = 42
