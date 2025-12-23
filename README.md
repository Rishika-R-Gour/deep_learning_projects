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

## Project 2: Binary Image Classification Using ConvNets

This project implements an **end-to-end computer vision pipeline** to classify food images into two categories: healthy salads and french fries. I collected and curated **several hundred real-world images** using the Bing Image Downloader, cleaned the dataset to remove noise and irrelevant samples, and organized the data into structured training, validation, and test splits stored locally within the Colab runtime.

I built a vanilla Convolutional Neural Network (ConvNet) from scratch using Keras image data generators and trained it on the curated dataset. Model evaluation showed **steady improvement in training and validation performance**, reaching ~90–95% validation accuracy with low final loss (~0.25) and a small, stable gap between curves, indicating **strong generalization** and **no evidence of overfitting**. This project demonstrates hands-on experience across the full computer vision workflow, from data collection and preprocessing to modeling and evaluation.

# Project 3: Socioeconomic Risk Classification Using Neural Networks

This project implements a neural network–based classification model to identify high-poverty communities using environmental, health, and socioeconomic indicators from the CalEnviroScreen 4.0 dataset.

The workflow includes data ingestion, feature selection, missing-value handling, and target engineering to convert poverty into a binary classification task. Input features include population, pollution exposure percentiles, and health and vulnerability indicators. Feature scaling was applied using training-only statistics to prevent data leakage.

A feedforward neural network was built using the Keras Sequential API with multiple dense layers, ReLU activations, dropout regularization, and early stopping. The model was trained using a 90/10 train–test split with a reproducible random seed.

On the held-out test set, the model achieved **~88% accuracy**, with strong balance across metrics **(Precision: 0.89, Recall: 0.84, F1-score: 0.87)**. Compared to a mean-only baseline accuracy of **~54%**, the neural network demonstrated substantial improvement, indicating meaningful learning and generalization.
