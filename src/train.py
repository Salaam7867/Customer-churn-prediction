# src/train.py

import numpy as np
import shap
import joblib
import os

os.makedirs("models", exist_ok=True)

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier


def train_xgb(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    param_grid = {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_grid,
        n_iter=15,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("Best Params:", search.best_params_)

    y_prob = best_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    print("Test ROC–AUC:", round(roc_auc, 4))

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    print("PR–AUC:", round(pr_auc, 4))

    # Threshold selection (Recall ≥ 0.85)
    thresholds = np.linspace(0.1, 0.9, 100)
    chosen_threshold = 0
    best_precision = 0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rec = recall_score(y_test, preds)
        prec = precision_score(y_test, preds)

        if rec >= 0.85 and prec > best_precision:
            best_precision = prec
            chosen_threshold = t

    print("Selected Threshold:", round(chosen_threshold, 3))

    y_pred = (y_prob >= chosen_threshold).astype(int)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred, digits=4))


    # ✅ SAVE MODEL ARTIFACTS
    joblib.dump(best_model, "models/xgb_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
    joblib.dump(chosen_threshold, "models/threshold.pkl")

    print("Model and threshold saved.")

    return best_model, chosen_threshold