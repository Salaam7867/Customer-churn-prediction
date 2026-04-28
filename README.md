# 🚀 Customer Churn Prediction (XGBoost – Business Optimized)

> End-to-end customer churn prediction pipeline using tuned XGBoost, recall-constrained threshold optimization, and SHAP explainability.

---

## 📌 Overview

This project builds a **production-ready churn prediction system** designed for real-world business deployment.

Instead of optimizing accuracy blindly, the model is designed to:

- 🎯 Maintain **Recall ≥ 85%** for churners  
- 📊 Maximize precision under business constraints  
- 🔍 Provide explainability using SHAP  
- 🚀 Be deployment-ready via Streamlit  

---

## 🧠 Problem Statement

Customer churn directly impacts revenue in subscription-based businesses.

The objective is to:

- Identify high-risk customers early  
- Reduce false negatives (missed churners)  
- Accept controlled false positives for retention strategy  

This is an **imbalanced classification problem (~26% churn rate)**.

---

## 📊 Final Model Performance (XGBoost)

| Metric | Value |
|--------|--------|
| ROC–AUC | **0.8438** |
| PR–AUC | **0.6645** |
| Recall (Churn) | **85.56%** |
| Precision (Churn) | **47.69%** |
| Selected Threshold | **0.399** |

### 🔎 Interpretation

- Captures majority of churners  
- Accepts manageable false positives  
- Suitable for proactive retention campaigns  

---

## ⚙️ Modeling Strategy

- Stratified train-test split  
- Class imbalance handling (`scale_pos_weight`)  
- Randomized hyperparameter tuning (5-fold CV)  
- Business-constrained threshold selection  
- SHAP explainability  
- Modular architecture  

---

## 🗂 Project Structure
customer-churn-prediction/
│
├── app/
│ └── app.py # Streamlit UI
│
├── models/
│ ├── xgb_model.pkl # Trained XGBoost model
│ ├── feature_columns.pkl # Feature schema
│ └── threshold.pkl # Business-selected threshold
│
├── src/
│ ├── preprocess.py # Data cleaning & feature prep
│ ├── train.py # Training pipeline
│ └── predict.py # Inference logic
│
├── main.py # Training entry point
├── requirements.txt
└── README.md


---

## 📦 Deployment

The model is deployed using **Streamlit**.

### 🔗 Live App link
https://customer-churn-prediction-salaam-73.streamlit.app/

The app:

- Accepts customer input  
- Applies preprocessing  
- Uses saved artifacts  
- Applies optimized threshold  
- Returns churn probability  

---

## 🔬 Explainability (SHAP)

Tree-based SHAP values are used for:

- Global feature importance  
- Model transparency  
- Business interpretation  

Top churn indicators typically include:

- Contract type  
- Tenure  
- Monthly charges  
- Internet service  

---

## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- SHAP  
- Streamlit  
- Git  

---

## 🖥 Run Locally

```bash
git clone https://github.com/Salaam7867/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app/app.py
