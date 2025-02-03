

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import (processed_dir, predictions_dir, reports_dir,
                    train_end, test_end, valid_end, 
                    pred_start, pred_end, RANDOM_SEED)

def plot_confusion_matrix(cm, labels, title='Confusion Matrix', out_path=None):
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()

def train_rf_model(ticker):

    symbol_reports_dir = os.path.join(reports_dir, ticker)
    os.makedirs(symbol_reports_dir, exist_ok=True)

    processed_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
    df = pd.read_csv(processed_file, parse_dates=['Date'], index_col='Date')
    
  
    df['Signal'] = np.where(df['Pct_Change'] > 0.02, 'Buy',
                    np.where(df['Pct_Change'] < -0.02, 'Sell', 'Hold'))
    
    features = ['Pct_Change', 'MA7', 'MA30', 'RSI', 'Momentum', 'Volatility']
    df.dropna(subset=features, inplace=True)
    
    train = df.loc[df.index < train_end]
    test  = df.loc[(df.index >= train_end) & (df.index < test_end)]
    valid = df.loc[(df.index >= test_end) & (df.index < valid_end)]
    
    X_train, y_train = train[features], train['Signal']
    X_test,  y_test  = test[features],  test['Signal']
    X_valid, y_valid = valid[features], valid['Signal']
    
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    
   
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cm_test = confusion_matrix(y_test, y_pred, labels=['Buy','Sell','Hold'])
    
    print(f"\n[RF - {ticker}] Test Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred))
    print(f"Test Confusion Matrix (Buy/Sell/Hold):\n{cm_test}")
    
    test_cm_path = os.path.join(symbol_reports_dir, f"{ticker}_rf_test_confusion_matrix.png")
    plot_confusion_matrix(cm_test, ['Buy','Sell','Hold'], 
                          title=f"{ticker} RF Test Confusion Matrix",
                          out_path=test_cm_path)
    

    yv_pred = model.predict(X_valid)
    val_acc = accuracy_score(y_valid, yv_pred)
    cm_valid = confusion_matrix(y_valid, yv_pred, labels=['Buy','Sell','Hold'])
    
    print(f"[RF - {ticker}] Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Confusion Matrix (Buy/Sell/Hold):\n{cm_valid}")
    
    valid_cm_path = os.path.join(symbol_reports_dir, f"{ticker}_rf_validation_confusion_matrix.png")
    plot_confusion_matrix(cm_valid, ['Buy','Sell','Hold'], 
                          title=f"{ticker} RF Validation Confusion Matrix",
                          out_path=valid_cm_path)
    
 
    feature_importances = model.feature_importances_
    feat_imp_sorted_idx = np.argsort(feature_importances)
    
    plt.figure(figsize=(6, 4))
    plt.barh([features[i] for i in feat_imp_sorted_idx], 
             [feature_importances[i] for i in feat_imp_sorted_idx])
    plt.title(f"{ticker} RF Feature Importances")
    plt.tight_layout()
    rf_imp_path = os.path.join(symbol_reports_dir, f"{ticker}_rf_feature_importances.png")
    plt.savefig(rf_imp_path)
    plt.close()
    
    return model, (test_acc, val_acc)

def generate_predictions_rf(ticker, model):

    processed_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
    if not os.path.exists(processed_file):
        return
    
    df = pd.read_csv(processed_file, parse_dates=['Date'], index_col='Date')
    
    features = ['Pct_Change', 'MA7', 'MA30', 'RSI', 'Momentum', 'Volatility']
    if 'Predicted_Signal' not in df.columns:
        df['Predicted_Signal'] = pd.Series(index=df.index, dtype="object")
    
    pred_df = df.loc[(df.index >= pred_start) & (df.index <= pred_end)]
    for date in pred_df.index:
        row = df.loc[[date], features]
        pred = model.predict(row)[0]
        df.loc[date, 'Predicted_Signal'] = pred
    
    os.makedirs(predictions_dir, exist_ok=True)
    pred_file = os.path.join(predictions_dir, f"{ticker}_predictions.csv")
    df.to_csv(pred_file, index_label='Date')
    print(f"[RF PREDICTIONS] {ticker} -> {pred_file}")

def backtest_strategy_rf(ticker, initial_cash=100000, fee_rate=0.001):

    pred_file = os.path.join(predictions_dir, f"{ticker}_predictions.csv")
    if not os.path.exists(pred_file):
        return None, None, None, None
    
    df = pd.read_csv(pred_file, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    
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
    
    if len(df) == 0:
        return None, None, None, None
    
    final_price = df['Open'].iloc[-1]
    final_value = cash + shares * final_price
    net_profit = final_value - initial_cash
    roi = net_profit / initial_cash
    
    print(f"\n[RF BACKTEST: {ticker}] Final Portfolio Value = ${final_value:,.2f}")
    print(f"[RF BACKTEST: {ticker}] Net Profit = ${net_profit:,.2f}")
    print(f"[RF BACKTEST: {ticker}] ROI = {roi:.2%}")
    
    return final_value, net_profit, roi, df

def daily_returns_for_strategy_rf(df_backtest, initial_cash=100000, fee_rate=0.001):

    df_backtest = df_backtest.copy()
    df_backtest['PortfolioValue'] = np.nan
    
    cash = initial_cash
    shares = 0
    
    if len(df_backtest) > 0:
        df_backtest.loc[df_backtest.index[0], 'PortfolioValue'] = cash
    
    for i in range(len(df_backtest) - 1):
        today = df_backtest.index[i]
        tomorrow = df_backtest.index[i + 1]
        
        signal_today = df_backtest.loc[today, 'Predicted_Signal']
        open_tomorrow = df_backtest.loc[tomorrow, 'Open']
        
        if signal_today == 'Buy' and shares == 0:
            cost = cash * (1 - fee_rate)
            shares = cost / open_tomorrow
            cash = 0
        elif signal_today == 'Sell' and shares > 0:
            proceeds = shares * open_tomorrow
            proceeds *= (1 - fee_rate)
            cash = proceeds
            shares = 0
        
        df_backtest.loc[df_backtest.index[i + 1], 'PortfolioValue'] = cash + shares * open_tomorrow
    
    df_backtest['DailyReturn'] = df_backtest['PortfolioValue'].pct_change()
    return df_backtest

def compute_beta(strategy_returns, benchmark_returns):
    cov_matrix = np.cov(strategy_returns, benchmark_returns)
    cov_value = cov_matrix[0, 1]
    var_bench = np.var(benchmark_returns)
    beta = cov_value / var_bench if var_bench != 0 else np.nan
    return beta
