# Customer Churn Prediction (Logistic Regression)

## Overview

This project builds an **end-to-end customer churn prediction pipeline** for a telecom dataset.
The focus is on **model comparison, business-driven evaluation, and a clean ML workflow**, not blind accuracy chasing.

Logistic Regression was selected as the **final model** after comparison with XGBoost, based on recall-focused performance.

---

## Problem Statement

Customer churn directly impacts revenue in subscription-based businesses.
The objective is to **identify customers likely to churn early**, enabling proactive retention strategies.

This is an **imbalanced classification problem**, where **missing churners (false negatives)** is more costly than flagging extra customers.

---

## Dataset

* Source: IBM Telco Customer Churn dataset
* Size: ~7,000 customers
* Target: `Churn` (Yes / No)
* Note: Dataset may be excluded from the repository for licensing reasons

---

## Key Challenges

* Class imbalance (~26% churn rate)
* Accuracy is misleading for churn problems
* Precision–recall trade-off
* Business-driven threshold selection

---

## Approach

### 1. Data Preprocessing

* Dropped non-informative identifier (`customerID`)
* Handled missing values in `TotalCharges`
* One-hot encoded categorical variables
* Scaled numerical features using `StandardScaler`
* Used stratified train–test split

---

### 2. Model Comparison

Two models were evaluated:

* **Logistic Regression**

  * Interpretable
  * Stable probabilities
  * Easy to deploy
* **XGBoost**

  * Strong non-linear learner
  * Better accuracy, slightly lower churn recall

Both models were evaluated using **ROC–AUC and recall-focused metrics**.

---

### 3. Final Model Selection

Although XGBoost achieved higher overall accuracy, **Logistic Regression was selected as the final model** because:

* Higher recall for churners
* Better alignment with retention-focused business goals
* Simpler and more interpretable decision logic

---

### 4. Threshold Tuning

Instead of using the default 0.5 probability threshold:

* Churn probabilities were analyzed
* A lower threshold was selected
* Goal: **maximize recall while keeping precision reasonable**

This reflects real-world churn prevention requirements.

---

## Evaluation Metrics

* Confusion Matrix
* Precision / Recall / F1-score
* ROC–AUC

Accuracy was **not** used as the primary decision metric.

---

## Results (Logistic Regression)

| Metric (Churn Class) | Value   |
| -------------------- | ------- |
| Recall               | ~90%    |
| Precision            | ~45–55% |
| ROC–AUC              | ~0.73   |

**Interpretation:**

* Most churners are correctly identified
* Higher false positives are accepted as a business trade-off
* Model is suitable for retention campaigns

---

## Repository Structure

* `comparison_model/`

  * `churn_model_comparison.ipynb` – Logistic Regression vs XGBoost analysis
  * `churn_logreg_model.pkl` – trained Logistic Regression model
  * `churn_xgb_model.pkl` – trained XGBoost model (comparison)
  * `scaler.pkl`, `feature_columns.pkl` – preprocessing artifacts
* `Trained_model.ipynb` – final training and evaluation notebook
* `app.py` – inference-ready prediction script
* `requirements.txt` – reproducible environment

---

## Business Interpretation

* High recall minimizes missed churners
* False positives are cheaper than lost customers
* Model decisions are transparent and explainable

---

## Limitations

* Linear model may miss complex non-linear interactions
* Precision can be improved with:

  * Better behavioral features
  * Temporal usage data
  * Ensemble approaches

---

## Future Improvements

* SHAP-based explainability
* LightGBM / XGBoost tuning
* Deployment via Streamlit or FastAPI
* Monitoring prediction drift over time

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Git & GitHub

---

## Author

**Mohd Abdul Salaam**
