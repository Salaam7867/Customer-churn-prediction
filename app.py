import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Churn Prediction")

BASE = os.path.dirname(__file__)

# Load artifacts
model = joblib.load(os.path.join(BASE, "churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
features = joblib.load(os.path.join(BASE, "feature_columns.pkl"))

def preprocess_input(df):
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["MonthlyCharges"] * df["tenure"]
    )

    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)

    df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        df[["tenure", "MonthlyCharges", "TotalCharges"]]
    )

    return df

st.title("Customer Churn Prediction")

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

    st.metric("Churn Probability", f"{prob:.2%}")
    st.success("Will Churn" if prob >= 0.35 else "Will Not Churn")
