Customer Churn Prediction (XGBoost – Business Optimized)

End-to-end customer churn prediction pipeline using tuned XGBoost, business-driven threshold selection, and SHAP-based explainability.

This project focuses on real-world deployment thinking, not just accuracy optimization.

Problem Statement

Customer churn directly impacts revenue in subscription-based businesses.

Objective:

Identify high-risk customers early

Maintain high recall for churners (≥ 85%)

Optimize precision under business constraints

Deploy a production-ready inference pipeline

This is an imbalanced binary classification problem (~26% churn rate).

Dataset

Source: IBM Telco Customer Churn Dataset

Records: ~7,000 customers

Target Variable: Churn (Yes / No)

Model Strategy

Instead of optimizing accuracy, the model was designed to:

Maximize churn recall (reduce false negatives)

Maintain acceptable precision

Select threshold based on business rule:

Recall ≥ 85%, maximize precision under this constraint

Final Model Performance (XGBoost)
Metric	Value
ROC–AUC	0.8438
PR–AUC	0.6645
Recall (Churn)	85.56%
Precision (Churn)	47.69%
Selected Threshold	0.399
Interpretation

Captures majority of churners

Accepts higher false positives for retention benefit

Suitable for proactive retention campaigns

Key Features

Stratified train-test split

Imbalance handling using scale_pos_weight

Randomized hyperparameter search (5-fold CV)

Business-constrained threshold optimization

SHAP explainability

Modular project structure

Deployment-ready artifacts

Project Structure
customer-churn-prediction/
│
├── app/
│   └── app.py                # Streamlit UI
│
├── models/
│   ├── xgb_model.pkl         # Trained XGBoost model
│   ├── feature_columns.pkl   # Feature schema
│   └── threshold.pkl         # Business-selected threshold
│
├── src/
│   ├── preprocess.py         # Data cleaning & feature prep
│   ├── train.py              # Training pipeline
│   └── predict.py            # Inference logic
│
├── main.py                   # Training entry point
├── requirements.txt
└── README.md
Deployment

The model is deployed using Streamlit.

Live App

(Insert your Streamlit link here)

The app:

Accepts user input

Applies preprocessing

Uses saved model artifacts

Applies optimized threshold

Outputs churn probability

Explainability

SHAP TreeExplainer is used for:

Global feature importance

Model transparency

Business interpretability

Top churn indicators typically include:

Contract type

Tenure

Monthly charges

Internet service type

Why XGBoost Over Logistic Regression?

Although Logistic Regression performed well:

XGBoost achieved higher ROC–AUC

Better PR–AUC under imbalance

More robust feature interactions

Improved recall–precision tradeoff

Final model selected based on business alignment.

Tech Stack

Python

Pandas

NumPy

Scikit-learn

XGBoost

SHAP

Streamlit

Git

How to Run Locally
git clone https://github.com/Salaam7867/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app/app.py
Author

Mohd Abdul Salaam
B.E. Computer Science Engineering
Aspiring Machine Learning / AI Engineer
