# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Telecom customer churn prediction with Revenue at Risk (RAR) analysis. Single Jupyter notebook (`code.ipynb`) implementing an end-to-end ML pipeline on a 7,043-row telecom dataset.

## Running the Project

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn shap
```

Open `code.ipynb` in Jupyter and run cells sequentially. The notebook reads `telcom_dataset.csv` (semicolon-separated, European decimal format) from the same directory.

## Architecture (Pipeline Stages in code.ipynb)

1. **Data Cleaning & Feature Engineering** — Parses European-format monetary columns, bucketizes tenure into 7 groups, applies binary and one-hot encoding (26 final columns)
2. **EDA** — Distributions, correlation heatmap, PCA (95% variance), K-Means clustering (k=3, cluster 0 holds 75% of churned customers)
3. **Train/Test Split & SMOTE** — 80/20 stratified split, SMOTE oversampling on training set only (balances 26.5% churn minority class)
4. **Model Training** — Logistic Regression (baseline), Random Forest (primary, n_estimators=1000, class_weight={0:1, 1:2}), XGBoost (GridSearchCV with 5-fold StratifiedKFold)
5. **Evaluation** — Confusion matrices, ROC/AUC curves, SHAP values, permutation importance
6. **Business Impact** — Revenue at Risk calculation: ~$744K/year from 969 at-risk customers (19.55% of total revenue)

## Key Technical Details

- **Target variable:** Binary churn (0/1), imbalanced (73.5% / 26.5%)
- **Top features by permutation importance:** numTechTickets, InternetService_No, Contract_Two_year, MonthlyCharges, DeviceProtection
- **CSV parsing:** Uses `sep=';'` and requires string-to-float conversion for monetary columns (comma decimals)
- **SMOTE is applied only to the training set** to avoid data leakage
