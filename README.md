# Smart Trading Signals v2

## 📌 Project Overview

**Signal Project v2** is a comprehensive financial market analysis and trading signal generation system leveraging multiple machine learning models. The project integrates various data processing, model training, backtesting, and visualization components into a streamlined pipeline. The models employed include:
- **Random Forest (RF)** for classification-based trading signals.
- **XGBoost (XGB)** for advanced regression-based forecasting.
- **Long Short-Term Memory (LSTM)** neural networks for deep learning-based sequence modeling.
- **Facebook Prophet** for time-series forecasting.

This repository is structured to facilitate financial data ingestion, preprocessing, feature engineering, model training, prediction generation, and result evaluation.

---

## 📂 Project Directory Structure

```
SignalProjectV2/
│   main.py                   # Main execution pipeline
│   requirements.txt          # Required dependencies
│
├── data/                      # Data directory
│   ├── raw/                   # Raw financial data
│   ├── processed/             # Processed datasets with feature engineering
│   ├── predictions/           # Model-generated predictions
│   ├── reports/               # Performance evaluation and backtesting reports
│
├── src/                       # Source code directory
│   ├── backtest.py            # Backtesting functions for strategy evaluation
│   ├── config.py              # Configuration and parameter settings
│   ├── data_download.py       # Data acquisition from Yahoo Finance
│   ├── feature_engineering.py # Feature extraction and transformation
│   ├── pipeline.py            # Core pipeline for execution
│   ├── train_rf.py            # Training and evaluation of the Random Forest model
│   ├── train_xgb.py           # Training and evaluation of the XGBoost model
│   ├── train_lstm.py          # Training and evaluation of the LSTM model
│   ├── train_prophet.py       # Training and evaluation of the Prophet model
│   ├── visualization.py       # Visualization utilities for analysis
```

---

## 📦 Dependencies and Technologies Used

This project utilizes various Python libraries for machine learning, data processing, and visualization. The main dependencies include:

- **Python 3.8+** (Primary programming language)
- **pandas, numpy** (Data manipulation and numerical computations)
- **yfinance** (Financial market data retrieval)
- **scikit-learn** (Machine learning tools and utilities)
- **xgboost** (Extreme Gradient Boosting for regression and classification)
- **tensorflow / keras** (Deep learning framework for LSTM model)
- **Prophet** (Time series forecasting tool by Facebook)
- **matplotlib, seaborn** (Data visualization)

To install dependencies, run:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### 1️⃣ Download Financial Data
```bash
python src/data_download.py
```
This script fetches market data from Yahoo Finance for predefined stock symbols.

### 2️⃣ Apply Feature Engineering
```bash
python src/feature_engineering.py
```
This step processes raw market data and generates relevant financial indicators.

### 3️⃣ Run the Complete Pipeline
```bash
python main.py
```
Executing this command trains all machine learning models, generates predictions, performs backtesting, and saves reports.

---

## 📊 Machine Learning Models Used

### 🟢 1. Random Forest (RF) - `train_rf.py`
- Classifies stock movements into **Buy, Sell, or Hold** signals.
- Trained using technical indicators and historical price changes.
- Evaluated via **confusion matrices and accuracy scores**.
- **Backtesting included** to analyze performance on past data.

### 🔵 2. XGBoost (XGB) - `train_xgb.py`
- Regression-based price forecasting model.
- Uses historical price movements and technical indicators.
- Evaluates performance with **MAE, RMSE, and R² metrics**.

### 🟡 3. Long Short-Term Memory (LSTM) - `train_lstm.py`
- Deep learning model for sequence prediction.
- Captures time-series dependencies and trends.
- Evaluated on **Mean Absolute Error (MAE) and RMSE**.

### 🔴 4. Prophet - `train_prophet.py`
- Bayesian-based time-series forecasting model.
- Predicts future stock prices with seasonal adjustments.
- Evaluated using **MAE, RMSE, and R²**.

---

## 🔄 Backtesting & Performance Evaluation

- **Strategy Backtesting (`backtest.py`)**
  - Tests trading strategies on historical data.
  - **ROI (Return on Investment) and net profit analysis**.
  - Computes **Beta values** for portfolio risk assessment.

---

## 📈 Data Visualization

Visualization is key to understanding model performance. The `visualization.py` script provides multiple plotting functions:

- `plot_time_series_with_predictions()` – Visualizes stock price trends and predicted buy/sell signals.
- `plot_bollinger_bands_with_predictions()` – Displays Bollinger Bands with buy/sell opportunities.
- `plot_xgb_feature_importances()` – Analyzes feature importance in the XGBoost model.
- `plot_lstm_training_loss()` – Monitors LSTM model training progress.
- ![TSLA_rf_test_confusion_matrix](https://github.com/user-attachments/assets/df8e8210-ea95-421a-8096-754291e2ea52)

![TSLA_prophet_forecast_full_pipeline](https://github.com/user-attachments/assets/3892f92e-d236-4444-abb2-c42cf44261a8)
![TSLA_rf_time_series_with_preds_from_pipeline](https://github.com/user-attachments/assets/d6f0db46-38cc-4099-815a-ea0322bcd8dd)
![TSLA_xgb_test_vs_pred_pipeline](https://github.com/user-attachments/assets/f08a710f-daad-443a-9683-a9396c186ac9)




