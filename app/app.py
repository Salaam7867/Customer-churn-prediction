import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Churn Prediction")

BASE = os.path.dirname(__file__)

# Load model artifacts
model = joblib.load(os.path.join(BASE, "..", "models", "xgb_model.pkl"))
features = joblib.load(os.path.join(BASE, "..", "models", "feature_columns.pkl"))
threshold = joblib.load(os.path.join(BASE, "..", "models", "threshold.pkl"))


def preprocess_input(df):
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["MonthlyCharges"] * df["tenure"]
    )

    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)

    return df


st.title("Customer Churn Prediction (XGBoost)")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

if st.button("Predict Churn"):

    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract
    }

    df = pd.DataFrame([data])
    X = preprocess_input(df)

    prob = model.predict_proba(X)[0][1]
    prediction = 1 if prob >= threshold else 0

    st.metric("Churn Probability", f"{prob:.2%}")

    if prediction == 1:
        st.error("High Risk: Likely to Churn")
    else:
        st.success("Low Risk: Likely to Stay")

    st.caption(f"Decision Threshold: {threshold:.2f}")