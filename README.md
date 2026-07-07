# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)

A complete end-to-end Machine Learning project to detect fraudulent credit card transactions using real world data from Kaggle.

## Problem Statement
Credit card fraud is a major problem in the financial industry. This project builds a system to identify fraudulent transactions where fraud cases are extremely rare (less than 0.2% of all transactions).

## Dataset
- Source: Kaggle Credit Card Fraud Detection
- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Size: 284,807 transactions
- Fraud cases: 492 (0.17%)

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Imbalanced-learn (SMOTE)
- Streamlit (Deployment)
- Matplotlib, Seaborn

## Project Pipeline
1. Exploratory Data Analysis (EDA)
2. Feature Scaling (StandardScaler)
3. Handling Class Imbalance (SMOTE)
4. Model Training (3 Models)
5. Model Evaluation (ROC-AUC, Confusion Matrix)
6. Deployment (Streamlit Web App)

## Models Used
| Model | Type |
|---|---|
| Logistic Regression | Baseline |
| XGBoost | Best Performer |
| Isolation Forest | Anomaly Detection |

## Results
- Fraud Detection Probability: 99.93%
- Evaluation Metrics: ROC-AUC, Precision-Recall, Confusion Matrix

## How to Run Locally
1. Clone this repository
2. Download dataset from Kaggle link above
3. Place creditcard.csv in project folder
4. Install dependencies:
   pip install -r requirements.txt
5. Run the app:
   streamlit run app.py

## Project Structure
fraud-detection/
├── app.py
├── xgb_model.pkl
├── creditcard_small.csv
├── requirements.txt
└── README.md
