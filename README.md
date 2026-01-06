# Customer Churn Prediction (Logistic Regression)

## Overview
This project predicts customer churn in a telecom dataset using a supervised
machine learning pipeline. The focus of the project is **model interpretability,
metric trade-offs, and production-ready workflow**, not just accuracy.

---

## Problem Statement
Customer churn directly impacts revenue in subscription-based businesses.
The goal is to **identify customers likely to churn early**, allowing retention
strategies to be applied.

This is an **imbalanced classification problem**, where missing churners
(False Negatives) is costlier than flagging extra customers.

---

## Dataset
- Source: IBM Telco Customer Churn dataset
- Rows: ~7,000 customers
- Target: `Churn` (Yes / No)
- Dataset is intentionally **excluded from the repository**

---

## Key Challenges
- Class imbalance (~26% churn rate)
- Precision–recall trade-off
- Over-optimistic accuracy metrics
- Business-driven evaluation (recall > accuracy)

---

## Approach

### 1. Data Preprocessing
- Removed non-informative identifiers (`customerID`)
- Handled missing values in `TotalCharges`
- One-hot encoded categorical features
- Scaled numerical features using `StandardScaler`

### 2. Model Selection
- Logistic Regression chosen for:
  - Interpretability
  - Probability-based decision control
  - Ease of deployment
- Class imbalance handled via `class_weight`

### 3. Threshold Tuning
Instead of default 0.5 threshold:
- Probabilities were used
- Decision threshold tuned manually
- Goal: **maximize recall while keeping precision reasonable**

### 4. Evaluation Metrics
- Confusion Matrix
- Precision / Recall / F1-score
- ROC–AUC

Accuracy was **not** used as the primary metric.

---

## Results

| Metric (Churn Class) | Value |
|---------------------|-------|
| Recall              | ~90%  |
| Precision           | ~45–55% |
| ROC–AUC             | ~0.73 |

**Interpretation:**
- Model successfully captures most churners
- Accepts higher false positives as a business trade-off
- Suitable for retention-focused strategies

---

## Business Interpretation
- High recall ensures fewer churners are missed
- False positives are acceptable compared to lost customers
- Model aligns with real-world churn prevention use cases

---

## Files in Repository
- `Log_reg2.ipynb` – training, evaluation, and analysis
- `app.py` – inference-ready prediction script
- `churn_model.pkl` – trained Logistic Regression model
- `scaler.pkl` – fitted feature scaler
- `feature_columns.pkl` – feature alignment for inference
- `requirements.txt` – reproducible environment

---

## Limitations
- Linear model may miss complex non-linear patterns
- Precision could be improved with:
  - Stronger behavioral features
  - Tree-based models
  - Temporal data

---

## Future Improvements
- Add SHAP-based feature explanations
- Compare with XGBoost / LightGBM
- Deploy via Streamlit or FastAPI
- Incorporate customer activity trends

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Git & GitHub

---

## Author
Mohd Abdul Salaam
