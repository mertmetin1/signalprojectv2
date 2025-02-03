
import os
import pandas as pd

from config import (symbols, raw_dir, processed_dir, reports_dir,
                    train_end, test_end, valid_end,
                    start_date, end_date, 
                    pred_start, pred_end)
from data_download import download_data
from feature_engineering import add_features
from train_rf import train_rf_model, generate_predictions_rf, backtest_strategy_rf, daily_returns_for_strategy_rf
from train_prophet import train_prophet_model
from train_xgb import train_xgb_model
from train_lstm import train_lstm_model


from visualization import (
    plot_confusion_matrix,
    plot_time_series_with_predictions,
    plot_bollinger_bands_with_predictions,
    plot_prophet_forecast,
    plot_prophet_test_period,
    plot_xgb_feature_importances,
    plot_xgb_test_results,
    plot_lstm_training_loss,
    plot_lstm_test_comparison
)

rf_scores = {}
prophet_results = {}
xgb_results = {}
lstm_results = {}

def main_pipeline():
  
    download_data()
    
   
    for ticker in symbols:
        raw_path = os.path.join(raw_dir, f"{ticker}_data.csv")
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path, parse_dates=['Date'], index_col='Date')
            df = add_features(df)
            processed_path = os.path.join(processed_dir, f"{ticker}_processed.csv")
            df.to_csv(processed_path, index_label='Date')
            print(f"[PROCESSED] {ticker} -> {processed_path}")
    
   
    for ticker in symbols:
        model, (test_acc, val_acc) = train_rf_model(ticker)
        if model:
            rf_scores[ticker] = (test_acc, val_acc)
            
           
            generate_predictions_rf(ticker, model)
            
          
            final_val, net_profit, roi, df_pred = backtest_strategy_rf(ticker)
            
          
            if df_pred is not None:    
                symbol_reports_dir = os.path.join(reports_dir, ticker)
                if not os.path.exists(symbol_reports_dir):
                    os.makedirs(symbol_reports_dir)
                
                ts_plot_path = os.path.join(symbol_reports_dir, f"{ticker}_rf_time_series_with_preds_from_pipeline.png")
                plot_time_series_with_predictions(
                    df_pred, 
                    ticker=ticker,
                    train_end=train_end,
                    test_end=test_end,
                    valid_end=valid_end,
                    out_path=ts_plot_path
                )
                
                bb_plot_path = os.path.join(symbol_reports_dir, f"{ticker}_rf_bollinger_bands_with_preds_from_pipeline.png")
                plot_bollinger_bands_with_predictions(
                    df_pred,
                    ticker=ticker,
                    period=20,
                    train_end=train_end,
                    test_end=test_end,
                    valid_end=valid_end,
                    out_path=bb_plot_path
                )
    
    for ticker in symbols:
        model, forecast, merged_df, (mae, rmse, r2) = train_prophet_model(ticker)
        if model:
            prophet_results[ticker] = (mae, rmse, r2)
            
            symbol_reports_dir = os.path.join(reports_dir, ticker)
            if not os.path.exists(symbol_reports_dir):
                os.makedirs(symbol_reports_dir)
            
            prophet_full_path = os.path.join(symbol_reports_dir, f"{ticker}_prophet_forecast_full_pipeline.png")
            plot_prophet_forecast(model, forecast, ticker, out_path=prophet_full_path)
            

            prophet_test_path = os.path.join(symbol_reports_dir, f"{ticker}_prophet_test_period_pipeline.png")
            plot_prophet_test_period(merged_df, ticker, out_path=prophet_test_path)
    

    for ticker in symbols:
        model, (mae, rmse, r2), test_data, y_pred = train_xgb_model(ticker)
        if model:
            xgb_results[ticker] = (mae, rmse, r2)
            

            symbol_reports_dir = os.path.join(reports_dir, ticker)
            if not os.path.exists(symbol_reports_dir):
                os.makedirs(symbol_reports_dir)
            
            feature_list = ['Close','Pct_Change','MA7','MA30','RSI','Momentum','Volatility']
            xgb_imp_path = os.path.join(symbol_reports_dir, f"{ticker}_xgb_feature_importances_pipeline.png")
            plot_xgb_feature_importances(model, feature_list, ticker, out_path=xgb_imp_path)
            

            xgb_test_plot_path = os.path.join(symbol_reports_dir, f"{ticker}_xgb_test_vs_pred_pipeline.png")
            dates = test_data['Date'].values
            y_test_vals = test_data['Target_Close'].values
            plot_xgb_test_results(dates, y_test_vals, y_pred, ticker, model_name="XGB", out_path=xgb_test_plot_path)
    

    for ticker in symbols:
        (model, 
         (mae, rmse, r2), 
         history, 
         test_data, 
         y_test_inv, 
         y_pred_inv, 
         seq_len) = train_lstm_model(ticker)
        
        if model:
            lstm_results[ticker] = (mae, rmse, r2)
            

            symbol_reports_dir = os.path.join(reports_dir, ticker)
            if not os.path.exists(symbol_reports_dir):
                os.makedirs(symbol_reports_dir)
            
            lstm_loss_path = os.path.join(symbol_reports_dir, f"{ticker}_lstm_loss_history_pipeline.png")
            plot_lstm_training_loss(history, ticker, out_path=lstm_loss_path)
            

            if len(test_data) > seq_len:
                date_vals = test_data['Date'].values[seq_len:]
                lstm_test_plot_path = os.path.join(symbol_reports_dir, f"{ticker}_lstm_test_vs_pred_pipeline.png")
                plot_lstm_test_comparison(date_vals, y_test_inv, y_pred_inv, ticker, out_path=lstm_test_plot_path)
    

    final_comparison = {}
    for sym in symbols:
        row = {}
     
        if sym in rf_scores:
            (rf_test_acc, rf_val_acc) = rf_scores[sym]
            row["RF_TestAcc"] = rf_test_acc
            row["RF_ValAcc"] = rf_val_acc
        else:
            row["RF_TestAcc"] = None
            row["RF_ValAcc"] = None

        if sym in prophet_results:
            (p_mae, p_rmse, p_r2) = prophet_results[sym]
            row["Prophet_MAE"]  = p_mae
            row["Prophet_RMSE"] = p_rmse
            row["Prophet_R2"]   = p_r2
        else:
            row["Prophet_MAE"]  = None
            row["Prophet_RMSE"] = None
            row["Prophet_R2"]   = None

        if sym in xgb_results:
            (x_mae, x_rmse, x_r2) = xgb_results[sym]
            row["XGB_MAE"]  = x_mae
            row["XGB_RMSE"] = x_rmse
            row["XGB_R2"]   = x_r2
        else:
            row["XGB_MAE"]  = None
            row["XGB_RMSE"] = None
            row["XGB_R2"]   = None
        

        if sym in lstm_results:
            (l_mae, l_rmse, l_r2) = lstm_results[sym]
            row["LSTM_MAE"]  = l_mae
            row["LSTM_RMSE"] = l_rmse
            row["LSTM_R2"]   = l_r2
        else:
            row["LSTM_MAE"]  = None
            row["LSTM_RMSE"] = None
            row["LSTM_R2"]   = None
        
        final_comparison[sym] = row
    
    final_df = pd.DataFrame.from_dict(final_comparison, orient='index')
    final_df.index.name = "Symbol"
    final_csv = os.path.join(reports_dir, "final_comparison_results.csv")
    final_df.to_csv(final_csv)
    print(f"\nAll model results are saved at   -> {final_csv}")
    print("\n=== Conclusion ===")
    print(final_df.head())

    print("\nPipeline Done!")

if __name__ == "__main__":
    main_pipeline()
