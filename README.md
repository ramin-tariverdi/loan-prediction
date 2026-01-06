# Loan Default Prediction & Model Evaluation

This project implements a complete data science pipeline to predict loan defaults using a Logit Regression model. The analysis covers data cleaning, exploratory data visualization, feature selection, and rigorous performance evaluation.

## Project Overview

### 1. Data Integrity & Exploration

The analysis is based on a financial dataset (`data.csv`) containing 1,000 entries and 16 unique features.

- **Pre-processing**: The dataset was verified to have no missing values or duplicate rows.
- **Feature Set**: The dataset includes financial indicators such as `Checking_amount`, `Credit_score`, `Saving_amount`, and demographic data like `Age` and `Gender`.
- **Exploratory Data Analysis (EDA)**: Histograms were generated for all numerical variables (Age, Amount, Saving Amount, etc.) to identify distributions and data variance.

### 2. Statistical Modeling (Logit Regression)

The project identifies key predictors of loan default using a Logistic Regression framework.

- **Model Optimization**: An initial model including all 16 features was refined by removing variables with p-values higher than 5% (backward elimination) to ensure statistical significance.
- **Significant Predictors**: Variables such as `Checking_amount`, `Credit_score`, `Saving_amount`, and `Age` were identified as highly significant predictors with P-values < 0.001.
- **Model Fit**: The final optimized model achieved a **Pseudo R-squared of 0.7237**.

### 3. Model Evaluation & Performance

The model was evaluated using a 70/30 train-test split to ensure its predictive power on unseen data.

- **F1-Score**: The model achieved an **F1-score of 0.878** on the test data, showing a strong balance between precision and recall.
- **Confusion Matrix (Test Data)**:
  - **True Positives**: 185 (Correctly identified defaults).
  - **False Positives**: 12 (Non-defaults incorrectly flagged as defaults).
  - **False Negatives**: 15 (Actual defaults missed by the model).
- **Conclusion**: The model demonstrates consistent performance across both training and test datasets with minimal variation.

## Technology Stack

- **Language**: Python 3.x
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `scipy`
- **Platform**: Google Colab / Jupyter Notebook

## How to Run

1. Ensure the dataset `data.csv` is in your working directory.
2. Install the necessary libraries:
   ```bash
   pip install pandas numpy matplotlib statsmodels scipy
   ```
3. Open and run all cells in `loan_prediction.ipynb` to reproduce the findings.
