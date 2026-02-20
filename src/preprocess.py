# src/preprocess.py

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["MonthlyCharges"] * df["tenure"]
    )

    return df


def prepare_features(df: pd.DataFrame):
    y = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["Churn"])

    cat_cols = df.select_dtypes(include="object").columns
    X = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return X, y