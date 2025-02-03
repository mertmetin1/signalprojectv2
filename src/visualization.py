

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels, title='Confusion Matrix', out_path=None):

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6, 5))
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


def plot_rf_feature_importances(feature_importances, feature_names, title="RF Feature Importances", out_path=None):

    sorted_idx = np.argsort(feature_importances)
    
    plt.figure(figsize=(6, 4))
    plt.barh([feature_names[i] for i in sorted_idx], 
             [feature_importances[i] for i in sorted_idx])
    plt.title(title)
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()


def plot_time_series_with_predictions(df, ticker, 
                                      train_end=None, test_end=None, valid_end=None,
                                      out_path=None):

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close', color='blue')
    
    # Örneğin MA7 / MA30 da varsa:
    if 'MA7' in df.columns:
        plt.plot(df.index, df['MA7'], label='MA7', color='orange', linestyle='--')
    if 'MA30' in df.columns:
        plt.plot(df.index, df['MA30'], label='MA30', color='green', linestyle='--')
    
    if 'Predicted_Signal' in df.columns:
        buy_points = df[df['Predicted_Signal'] == 'Buy']
        sell_points = df[df['Predicted_Signal'] == 'Sell']
        hold_points = df[df['Predicted_Signal'] == 'Hold']
        plt.scatter(buy_points.index, buy_points['Close'], color='green', marker='^', s=100, label='Buy')
        plt.scatter(sell_points.index, sell_points['Close'], color='red', marker='v', s=100, label='Sell')
        plt.scatter(hold_points.index, hold_points['Close'], color='gray', marker='.', s=50, label='Hold')
    
    if train_end is not None:
        plt.axvline(x=train_end, color='red', linestyle='--', label='Train End')
    if test_end is not None:
        plt.axvline(x=test_end, color='purple', linestyle='--', label='Test End')
    if valid_end is not None:
        plt.axvline(x=valid_end, color='brown', linestyle='--', label='Valid End')
    
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{ticker} Time Series & (Optional) Predicted Signals")
    plt.legend()
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()


def plot_bollinger_bands_with_predictions(df, ticker, period=20,
                                          train_end=None, test_end=None, valid_end=None,
                                          out_path=None):

    data = df.copy()
    
    ma_label = f"MA_{period}"
    std_label = f"STD_{period}"
    data[ma_label] = data['Close'].rolling(period).mean()
    data[std_label] = data['Close'].rolling(period).std()
    
    data['UpperBand'] = data[ma_label] + 2 * data[std_label]
    data['LowerBand'] = data[ma_label] - 2 * data[std_label]
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, data[ma_label], label=f"{period}-day MA", color='green', linestyle='--')
    plt.plot(data.index, data['UpperBand'], label='Upper Band', color='red', linestyle='--')
    plt.plot(data.index, data['LowerBand'], label='Lower Band', color='red', linestyle='--')
    
    plt.fill_between(data.index, data['UpperBand'], data['LowerBand'], color='grey', alpha=0.2)
    
    if 'Predicted_Signal' in data.columns:
        buy_points = data[data['Predicted_Signal'] == 'Buy']
        sell_points = data[data['Predicted_Signal'] == 'Sell']
        hold_points = data[data['Predicted_Signal'] == 'Hold']
        plt.scatter(buy_points.index, buy_points['Close'], color='green', marker='^', s=100, label='Buy')
        plt.scatter(sell_points.index, sell_points['Close'], color='red', marker='v', s=100, label='Sell')
        plt.scatter(hold_points.index, hold_points['Close'], color='gray', marker='.', s=50, label='Hold')
    
    if train_end is not None:
        plt.axvline(x=train_end, color='red', linestyle='--', label='Train End')
    if test_end is not None:
        plt.axvline(x=test_end, color='purple', linestyle='--', label='Test End')
    if valid_end is not None:
        plt.axvline(x=valid_end, color='brown', linestyle='--', label='Valid End')
    
    plt.title(f"{ticker} Bollinger Bands (Period={period}) + Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()


def plot_prophet_forecast(model, forecast, ticker, out_path=None):

    fig, ax = plt.subplots(figsize=(12, 6))
    model.plot(forecast, ax=ax)
    plt.title(f"{ticker} - Prophet Full Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()


def plot_prophet_test_period(merged_df, ticker, out_path=None):

    plt.figure(figsize=(12, 6))
    plt.plot(merged_df['ds'], merged_df['y'], label='Actual', color='blue')
    plt.plot(merged_df['ds'], merged_df['yhat'], label='Predicted', color='red', linestyle='--')
    plt.title(f"{ticker} - Prophet Test Period")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()


def plot_xgb_feature_importances(model, feature_names, ticker, out_path=None):

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    
    plt.figure(figsize=(6, 4))
    plt.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx])
    plt.title(f"{ticker} XGBoost Feature Importances")
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()


def plot_xgb_test_results(dates, y_true, y_pred, ticker, model_name="XGB", out_path=None):

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='Actual', color='blue')
    plt.plot(dates, y_pred, label='Predicted', color='red', linestyle='--')
    plt.title(f"{ticker} - {model_name} Test Comparison")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()


def plot_lstm_training_loss(history, ticker, out_path=None):

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{ticker} - LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()


def plot_lstm_test_comparison(dates, y_true, y_pred, ticker, out_path=None):

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='Actual', color='blue')
    plt.plot(dates, y_pred, label='Predicted', color='red', linestyle='--')
    plt.title(f"{ticker} - LSTM Test Comparison")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()
