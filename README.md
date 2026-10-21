# Exoplanet Classification & Temperature Prediction

A MATLAB-based data mining project that applies Support Vector Machine (SVM) for exoplanet classification and Principal Component Regression (PCR) for equilibrium temperature prediction using the Kepler dataset. The project covers data preprocessing, model evaluation, and habitability analysis.

## Project Objectives

- Classify exoplanets into Confirmed, Candidate, and False Positive categories.
- Predict exoplanet equilibrium temperatures using Principal Component Regression (PCR).
- Evaluate classification and regression model performance.
- Analyze exoplanet habitability indicators using calculated habitability scores.

## Dataset

This project uses the [Kepler Exoplanet Search Results dataset](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) provided on Kaggle.

The dataset contains planetary and stellar parameters used for exoplanet classification and equilibrium temperature prediction.

## Methods

### 1. Exoplanet Classification

- **Algorithm:** Linear Support Vector Machine (SVM)
- **Target Classes:** Confirmed, Candidate, False Positive
- **Data Preprocessing:** Z-score normalization
- **Evaluation Metrics:** Accuracy, F1-score, stratified 5-fold cross-validation

### 2. Equilibrium Temperature Prediction

- **Algorithm:** Principal Component Regression (PCR)
- **Target Variable:** Exoplanet equilibrium temperature
- **Evaluation Metrics:** R-squared (R²), Root Mean Squared Error (RMSE)

### 3. Habitability Analysis

Analyzes habitability scores to identify potentially habitable exoplanets among confirmed planets in the dataset.

## Technologies & Tools

- **Programming Language:** MATLAB
- **Classification:** Support Vector Machine (SVM)
- **Regression:** Principal Component Regression (PCR)
- **Preprocessing:** Z-score normalization
- **Model Evaluation:** Accuracy, F1-score, R², RMSE, cross-validation
