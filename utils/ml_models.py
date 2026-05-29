"""
Machine Learning models for Flight Delay Prediction.
Trains RandomForest, GradientBoosting, and Logistic Regression classifiers.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


FEATURE_COLS = [
    "MONTH", "DAY_OF_WEEK", "DAY_OF_MONTH",
    "CRS_DEP_TIME", "DISTANCE", "CRS_ELAPSED_TIME",
    "OP_CARRIER_ENC", "ORIGIN_ENC", "DEST_ENC", "WEATHER_ENC",
]

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class FlightDelayPredictor:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importances = {}
        self.metrics = {}
        self.X_test = None
        self.y_test = None

    def preprocess(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        cat_cols = {
            "OP_CARRIER": "OP_CARRIER_ENC",
            "ORIGIN": "ORIGIN_ENC",
            "DEST": "DEST_ENC",
            "WEATHER_CONDITION": "WEATHER_ENC",
        }
        for col, enc_col in cat_cols.items():
            if col not in df.columns:
                df[enc_col] = 0
                continue
            if fit:
                le = LabelEncoder()
                df[enc_col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders.get(col)
                if le:
                    df[enc_col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    df[enc_col] = 0
        return df

    def train(self, df: pd.DataFrame):
        df = self.preprocess(df, fit=True)

        # Filter out cancelled flights for delay prediction
        df_active = df[df["CANCELLED"] == 0].copy()
        df_active = df_active.dropna(subset=["IS_DELAYED"])

        X = df_active[FEATURE_COLS].fillna(0)
        y = df_active["IS_DELAYED"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.X_test = X_test
        self.y_test = y_test

        model_configs = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=5,
                random_state=42, n_jobs=-1, class_weight="balanced"
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42
            ),
            "Logistic Regression": Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")),
            ]),
        }

        for name, model in model_configs.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            prec, rec, _ = precision_recall_curve(y_test, y_prob)

            self.models[name] = model
            self.metrics[name] = {
                "accuracy": acc,
                "auc": auc,
                "report": report,
                "confusion_matrix": cm,
                "roc": (fpr, tpr),
                "pr_curve": (prec, rec),
            }

            # Feature importances (only for tree-based models)
            if hasattr(model, "feature_importances_"):
                self.feature_importances[name] = dict(
                    zip(FEATURE_COLS, model.feature_importances_)
                )

        self.is_trained = True
        return self.metrics

    def predict(self, input_dict: dict, model_name: str = "Random Forest") -> dict:
        """Predict delay for a single flight."""
        if not self.is_trained:
            return {"error": "Model not trained yet"}

        row = pd.DataFrame([input_dict])
        row = self.preprocess(row, fit=False)

        # Ensure all feature cols exist
        for col in FEATURE_COLS:
            if col not in row.columns:
                row[col] = 0

        X = row[FEATURE_COLS].fillna(0)
        model = self.models.get(model_name)
        if model is None:
            return {"error": f"Model {model_name} not found"}

        prob = model.predict_proba(X)[0][1]
        pred = int(prob >= 0.5)

        delay_minutes = 0
        if pred == 1:
            # Estimate delay in minutes based on probability
            delay_minutes = int(15 + prob * 120)

        return {
            "is_delayed": pred,
            "delay_probability": round(float(prob), 4),
            "estimated_delay_minutes": delay_minutes,
            "risk_level": _risk_label(prob),
        }

    def save(self, path: str = MODELS_DIR):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.models, os.path.join(path, "models.pkl"))
        joblib.dump(self.encoders, os.path.join(path, "encoders.pkl"))
        joblib.dump(self.metrics, os.path.join(path, "metrics.pkl"))
        joblib.dump(self.feature_importances, os.path.join(path, "feature_importances.pkl"))

    def load(self, path: str = MODELS_DIR):
        self.models = joblib.load(os.path.join(path, "models.pkl"))
        self.encoders = joblib.load(os.path.join(path, "encoders.pkl"))
        self.metrics = joblib.load(os.path.join(path, "metrics.pkl"))
        self.feature_importances = joblib.load(os.path.join(path, "feature_importances.pkl"))
        self.is_trained = True


def _risk_label(prob: float) -> str:
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Moderate"
    elif prob < 0.75:
        return "High"
    else:
        return "Very High"
 
