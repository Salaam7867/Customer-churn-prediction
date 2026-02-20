# src/predict.py

import pandas as pd


def preprocess_input(df, feature_columns):
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def predict(model, df, threshold):
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, probs