
# Phone Price Prediction Model - README

This repository contains code and methods for predicting phone prices using an **XGBoost Regressor**. The project involves data cleaning, preprocessing, exploratory data analysis (EDA), feature engineering, model training with hyperparameter tuning, and evaluation.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Preparation](#data-preparation)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Modeling and Hyperparameter Tuning](#modeling-and-hyperparameter-tuning)
6. [Evaluation and Analysis](#evaluation-and-analysis)
7. [Dependencies](#dependencies)
8. [Instructions to Run](#instructions-to-run)

---

### Project Overview

The primary objective of this project is to predict the **price of phones** in USD based on their technical specifications and other features. The model leverages a dataset containing phone information and prices, cleans and preprocesses the data, and applies **XGBoost** regression with parameter tuning to achieve high accuracy.

---

### Data Preparation

1. **Load Dataset**:
   - The data is loaded from a CSV file containing phone specifications and prices.

2. **Data Inspection**:
   - Initial inspection includes viewing the first few rows, statistical summary, data types, and missing values.

3. **Data Cleaning**:
   - Columns like `price_USD` and `price` are converted to numeric types, handling errors by replacing invalid values with `NaN`.
   - Missing values in `price_USD` are dropped, while other columns (e.g., `storage`, `ram`, `Weight`) are filled with median values.

4. **Data Type Conversion**:
   - Columns like `Launch` (date) and `Foldable` (binary) are converted to appropriate formats for better handling during analysis.

---

### Exploratory Data Analysis (EDA)

1. **Distribution of Phone Prices**:
   - A histogram displays the distribution of `price_USD` to understand the price spread.

2. **Price Distribution by Price Range**:
   - A boxplot of `price_USD` by `price_range` visually indicates how price varies across predefined categories.

3. **Average Price by Brand**:
   - A bar chart shows the average price per brand, providing insights into pricing trends by brand.

4. **Correlation Matrix**:
   - A heatmap of correlations among numerical features helps identify relationships between variables.

---

### Feature Engineering

1. **Additional Features**:
   - New features like `Launch_Year` (extracted from `Launch`) are created to leverage historical information.
   - Categorical variables (`Display_Type`, `Chipset`) are converted into dummy variables for model compatibility.

2. **Feature Selection**:
   - The final feature set includes `storage`, `ram`, `PPI_Density`, `Launch_Year`, `Weight`, and dummies from categorical columns.

---

### Modeling and Hyperparameter Tuning

1. **Splitting Data**:
   - The data is split into training and test sets with an 80-20 split.

2. **Model Selection**:
   - **XGBoost Regressor** is chosen as the model for prediction due to its efficiency and performance with structured data.

3. **Hyperparameter Tuning**:
   - A `RandomizedSearchCV` is implemented with a parameter grid to find optimal model parameters. This includes tuning:
     - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`.
   - The `RandomizedSearchCV` output is stored, and the best estimator is used for prediction.

---

### Evaluation and Analysis

1. **Model Performance Metrics**:
   - **RMSE (Root Mean Squared Error)** and **R² Score** are calculated to assess model performance on the test set.

2. **Feature Importance**:
   - The top 20 important features for predicting phone prices are visualized using a bar plot, highlighting key contributors to the model's decision-making.

3. **Predicted vs. Actual Prices Plot**:
   - A scatter plot compares predicted prices to actual prices, with a reference line indicating perfect predictions.

---

### Dependencies

- Python Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `xgboost`, `joblib`
- Make sure to install the libraries with:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
  ```

---

### Instructions to Run

1. **Prepare Data**:
   - Ensure the dataset file `processed_data news.csv` is in the same directory or update `file_path` to the correct path.

2. **Run the Code**:
   - Execute each section sequentially in a Jupyter Notebook or run the entire script if using a script-based environment.

3. **Review Output**:
   - Inspect EDA visuals, model evaluation metrics, and the feature importance plot for insights into phone pricing trends.

---
