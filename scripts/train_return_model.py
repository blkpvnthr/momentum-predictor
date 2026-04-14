from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


# =========================================================
# PATHS
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

DATA_PATH = PROJECT_ROOT / "outputs" / "training" / "labeled_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "return_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "return_model.joblib"
METRICS_PATH = OUTPUT_DIR / "return_model_metrics.json"
SCORED_PATH = OUTPUT_DIR / "return_model_scored.csv"


# =========================================================
# CONFIG
# =========================================================
TIMEZONE = "America/New_York"
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15

TRADE_THRESHOLDS = [0.0001, 0.0002, 0.0003, 0.0005]


# =========================================================
# LOAD
# =========================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

required_cols = [
    "timestamp",
    "pred_return_5m",
    "pred_return_15m",
    "pred_return_30m",
    "actual_return_15m",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(TIMEZONE)
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


# =========================================================
# FEATURE BUILD
# Works from labeled_predictions.csv, not raw close data
# =========================================================
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["minutes_from_open"] = (df["hour"] * 60 + df["minute"]) - (9 * 60 + 30)
df["minutes_to_close"] = (16 * 60) - (df["hour"] * 60 + df["minute"])

df["pred_abs_5m"] = df["pred_return_5m"].abs()
df["pred_abs_15m"] = df["pred_return_15m"].abs()
df["pred_abs_30m"] = df["pred_return_30m"].abs()

df["pred_spread_5_15"] = df["pred_return_15m"] - df["pred_return_5m"]
df["pred_spread_15_30"] = df["pred_return_30m"] - df["pred_return_15m"]

df["pred_sign_5m"] = np.sign(df["pred_return_5m"])
df["pred_sign_15m"] = np.sign(df["pred_return_15m"])
df["pred_sign_30m"] = np.sign(df["pred_return_30m"])

if "confidence" in df.columns:
    df["confidence_x_pred_15m"] = df["confidence"] * df["pred_return_15m"]
    df["confidence_x_abs_pred_15m"] = df["confidence"] * df["pred_abs_15m"]

if "signal" in df.columns:
    signal_map = {"NO_TRADE": 0, "LONG": 1, "SHORT": -1}
    df["signal_num"] = df["signal"].map(signal_map).fillna(0)

# Target from aligned realized outcome
df["target_return"] = df["actual_return_15m"]

feature_cols = [
    "pred_return_5m",
    "pred_return_15m",
    "pred_return_30m",
    "pred_abs_5m",
    "pred_abs_15m",
    "pred_abs_30m",
    "pred_spread_5_15",
    "pred_spread_15_30",
    "pred_sign_5m",
    "pred_sign_15m",
    "pred_sign_30m",
    "hour",
    "minute",
    "day_of_week",
    "minutes_from_open",
    "minutes_to_close",
]

optional_cols = [
    "confidence",
    "confidence_x_pred_15m",
    "confidence_x_abs_pred_15m",
    "signal_num",
]

feature_cols.extend([c for c in optional_cols if c in df.columns])

df = df.dropna(subset=["target_return"]).copy()
df = df.dropna(subset=feature_cols).copy()

if len(df) < 1000:
    raise ValueError(f"Not enough rows after preprocessing: {len(df)}")

print(f"Rows after preprocessing: {len(df)}")
print(f"Using {len(feature_cols)} features")
print(feature_cols)


# =========================================================
# TIME SPLIT
# =========================================================
n = len(df)
train_end = int(n * TRAIN_SPLIT)
val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

X_train = train_df[feature_cols]
y_train = train_df["target_return"]

X_val = val_df[feature_cols]
y_val = val_df["target_return"]

X_test = test_df[feature_cols]
y_test = test_df["target_return"]

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


# =========================================================
# MODEL
# =========================================================
sample_weight = np.clip(train_df["target_return"].abs() / 0.001, 1.0, 10.0)

pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)


# =========================================================
# EVALUATION
# =========================================================
val_pred = pipeline.predict(X_val)
test_pred = pipeline.predict(X_test)

def evaluate_split(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "corr": float(pd.Series(y_true).corr(pd.Series(y_pred))),
    }

    for thr in TRADE_THRESHOLDS:
        take = y_pred >= thr
        rows = int(take.sum())

        if rows > 0:
            mean_ret = float(y_true[take].mean())
            hit_rate = float((y_true[take] > 0).mean())
            robust_score = float(mean_ret * np.sqrt(rows))
        else:
            mean_ret = np.nan
            hit_rate = np.nan
            robust_score = np.nan

        out[f"thr_{thr}_rows"] = rows
        out[f"thr_{thr}_take_rate"] = float(take.mean())
        out[f"thr_{thr}_mean_return"] = mean_ret
        out[f"thr_{thr}_hit_rate"] = hit_rate
        out[f"thr_{thr}_robust_score"] = robust_score

    return out

val_metrics = evaluate_split("val", y_val, val_pred)
test_metrics = evaluate_split("test", y_test, test_pred)

# choose threshold from validation by robust score
best_threshold = max(
    TRADE_THRESHOLDS,
    key=lambda t: val_metrics.get(f"thr_{t}_robust_score", float("-inf"))
)

test_take = test_pred >= best_threshold
test_taken = test_df.loc[test_take].copy()

metrics = {
    "train_rows": int(len(train_df)),
    "val_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "best_threshold_from_val": float(best_threshold),
    "val_mae": val_metrics["mae"],
    "val_rmse": val_metrics["rmse"],
    "val_corr": val_metrics["corr"],
    "test_mae": test_metrics["mae"],
    "test_rmse": test_metrics["rmse"],
    "test_corr": test_metrics["corr"],
    "test_rows_taken": int(len(test_taken)),
    "test_take_rate": float(test_take.mean()),
    "test_mean_return_taken": float(test_taken["target_return"].mean()) if len(test_taken) else np.nan,
    "test_hit_rate_taken": float((test_taken["target_return"] > 0).mean()) if len(test_taken) else np.nan,
    "validation_threshold_metrics": val_metrics,
    "test_threshold_metrics": test_metrics,
}


# =========================================================
# SAVE SCORED DATA
# =========================================================
full_pred = pipeline.predict(df[feature_cols])
df["predicted_return_15m"] = full_pred
df["take_trade"] = (df["predicted_return_15m"] >= best_threshold).astype(int)
df["realized_pnl_if_taken"] = np.where(df["take_trade"] == 1, df["target_return"], 0.0)

save_cols = [
    "timestamp",
    "target_return",
    "predicted_return_15m",
    "take_trade",
    "realized_pnl_if_taken",
    *feature_cols,
]
scored_df = df[save_cols].copy()

joblib.dump(
    {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "best_threshold": best_threshold,
        "target_col": "target_return",
    },
    MODEL_PATH,
)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

scored_df.to_csv(SCORED_PATH, index=False)

print(f"Saved model to: {MODEL_PATH}")
print(f"Saved metrics to: {METRICS_PATH}")
print(f"Saved scored data to: {SCORED_PATH}")
print("\n=== METRICS ===")
for k, v in metrics.items():
    if k in {"validation_threshold_metrics", "test_threshold_metrics"}:
        continue
    print(f"{k}: {v}")