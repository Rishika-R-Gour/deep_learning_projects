# deep_learning_projects
## Project 1: Forecasting Daily Energy Production Using Weather Data

This project focuses on forecasting **daily energy production** using historical
generation data and daily weather variables. The goal was to predict **all days in 2025**
using only data available before January 1, 2025, closely simulating a real-world
energy forecasting scenario.

The workflow includes data cleaning, exploratory analysis, and **time-aware feature
engineering**, such as lagged variables, rolling averages, and calendar features.
Deep learning models including **DNN, LSTM, and GRU** were trained using a strict
time-based train–validation–test split to prevent data leakage.

The final model achieved strong performance on the 2025 holdout set:
- **MAPE:** 6.22%
- **MAE:** 18,839
- **RMSE:** 23,194

Results were analyzed to understand how weather patterns influence energy production,
and insights were translated into practical implications for energy planning and
operational decision-making.
