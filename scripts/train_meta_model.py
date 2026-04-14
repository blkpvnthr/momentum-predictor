from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# PATHS
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

INPUT_PATH = PROJECT_ROOT / "outputs" / "training" / "labeled_predictions.csv"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "meta_model.joblib"
METRICS_PATH = MODEL_DIR / "meta_model_metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "meta_model_scored.csv"


# =========================================================
# CONFIG
# =========================================================
TARGET_COL = "target_take_trade"
MIN_ABS_PRED_RETURN = 0.0
TEST_SIZE_FRACTION = 0.2


# =========================================================
# LOAD
# =========================================================
df = pd.read_csv(INPUT_PATH)

required = [
    "timestamp",
    "signal",
    "pred_return_5m",
    "pred_return_15m",
    "pred_return_30m",
    "confidence",
    "actual_return_15m",
    "pnl_15m",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
df = df.dropna(subset=["timestamp"]).copy()

# LONG-only for now, since your short side was weak
df = df[df["signal"] == "LONG"].copy()

# Optional trade-strength filter
df = df[df["pred_return_15m"].abs() >= MIN_ABS_PRED_RETURN].copy()

if len(df) < 100:
    raise ValueError(f"Not enough rows after filtering: {len(df)}")


# =========================================================
# FEATURE ENGINEERING
# =========================================================
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["minutes_from_open"] = (df["hour"] * 60 + df["minute"]) - (9 * 60 + 30)

df["confidence_abs_pred_15m"] = df["pred_return_15m"].abs()
df["pred_sign_15m"] = np.sign(df["pred_return_15m"])
df["pred_return_spread_5_15"] = df["pred_return_15m"] - df["pred_return_5m"]
df["pred_return_spread_15_30"] = df["pred_return_30m"] - df["pred_return_15m"]

# Binary target: should we take this trade?
df[TARGET_COL] = (df["pnl_15m"] > 0).astype(int)

feature_cols_numeric = [
    "pred_return_5m",
    "pred_return_15m",
    "pred_return_30m",
    "confidence",
    "confidence_abs_pred_15m",
    "pred_sign_15m",
    "pred_return_spread_5_15",
    "pred_return_spread_15_30",
    "hour",
    "minute",
    "minutes_from_open",
]

feature_cols_categorical = [
    "signal",
]

all_features = feature_cols_numeric + feature_cols_categorical

df = df.dropna(subset=[TARGET_COL]).copy()


# =========================================================
# TIME-BASED TRAIN/TEST SPLIT
# =========================================================
df = df.sort_values("timestamp").reset_index(drop=True)

split_idx = int(len(df) * (1 - TEST_SIZE_FRACTION))
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

X_train = train_df[all_features]
y_train = train_df[TARGET_COL]

X_test = test_df[all_features]
y_test = test_df[TARGET_COL]

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError("Train/test split produced empty partition.")


# =========================================================
# PIPELINE
# =========================================================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_numeric),
        ("cat", categorical_transformer, feature_cols_categorical),
    ]
)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# =========================================================
# TRAIN
# =========================================================
pipeline.fit(X_train, y_train)

test_probs = pipeline.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

metrics = {
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "base_positive_rate_test": float(y_test.mean()),
    "accuracy": float(accuracy_score(y_test, test_preds)),
    "precision": float(precision_score(y_test, test_preds, zero_division=0)),
    "recall": float(recall_score(y_test, test_preds, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else None,
    "classification_report": classification_report(y_test, test_preds, zero_division=0),
}

# Score full dataset
full_probs = pipeline.predict_proba(df[all_features])[:, 1]
df["meta_prob_take_trade"] = full_probs
df["meta_take_trade"] = (df["meta_prob_take_trade"] >= 0.5).astype(int)

# Save outputs
joblib.dump(
    {
        "pipeline": pipeline,
        "feature_cols_numeric": feature_cols_numeric,
        "feature_cols_categorical": feature_cols_categorical,
        "all_features": all_features,
        "target_col": TARGET_COL,
    },
    MODEL_PATH,
)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

df.to_csv(PREDICTIONS_PATH, index=False)

print(f"Saved model to: {MODEL_PATH}")
print(f"Saved metrics to: {METRICS_PATH}")
print(f"Saved scored dataset to: {PREDICTIONS_PATH}")
print("\n=== METRICS ===")
for k, v in metrics.items():
    print(f"{k}: {v}")