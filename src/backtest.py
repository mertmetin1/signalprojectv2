

import numpy as np
import pandas as pd
import os

def backtest_rf_strategy(df, initial_cash=100000, fee_rate=0.001):

    cash = initial_cash
    shares = 0
    
    for i in range(len(df) - 1):
        today = df.index[i]
        tomorrow = df.index[i + 1]
        signal_today = df.loc[today, 'Predicted_Signal']
        open_price_tomorrow = df.loc[tomorrow, 'Open']
        
        if signal_today == 'Buy':
            if shares == 0 and cash > 0:
                cost = cash * (1 - fee_rate)
                shares = cost / open_price_tomorrow
                cash = 0
        elif signal_today == 'Sell':
            if shares > 0:
                proceeds = shares * open_price_tomorrow
                proceeds *= (1 - fee_rate)
                cash = proceeds
                shares = 0
    
    final_price = df['Open'].iloc[-1]
    final_value = cash + shares * final_price
    net_profit = final_value - initial_cash
    roi = net_profit / initial_cash
    return final_value, net_profit, roi

def compute_beta(strategy_returns, bench_returns):
    cov_matrix = np.cov(strategy_returns, bench_returns)
    cov_value = cov_matrix[0,1]
    var_bench = np.var(bench_returns)
    beta = cov_value / var_bench if var_bench != 0 else np.nan
    return beta
