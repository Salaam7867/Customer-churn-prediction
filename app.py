from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

# Load saved artifacts
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("feature_columns.pkl")

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

@app.post("/predict")
def predict_churn(data: dict):
    df = pd.DataFrame([data])
    X = preprocess_input(df)
    prob = model.predict_proba(X)[0][1]

    return {
        "churn_probability": float(prob),
        "will_churn": bool(prob >= 0.35)
    }
