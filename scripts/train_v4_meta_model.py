from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


# =========================================================
# PATHS
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

INPUT_PATH = PROJECT_ROOT / "outputs" / "v2_return_model" / "v2_return_model_scored.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v4_meta_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "v4_meta_model.joblib"
METRICS_PATH = OUTPUT_DIR / "v4_meta_model_metrics.json"
SCORED_PATH = OUTPUT_DIR / "v4_meta_model_scored.csv"


# =========================================================
# CONFIG
# =========================================================
TIMEZONE = "America/New_York"

TEST_SIZE_FRACTION = 0.20
VALIDATION_SIZE_FRACTION = 0.15

# only train meta model on rows the base model was at least somewhat interested in
BASE_PRED_MIN = 0.0

# meta target: profitable after costs
COST_BUFFER = 0.0001

MIN_ROWS_AT_THRESHOLD = 50
META_THRESHOLD_QUANTILES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]


# =========================================================
# HELPERS
# =========================================================
def require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def parse_timestamp_series(series: pd.Series, timezone: str) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_convert(timezone)


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    corr = pd.Series(a).corr(pd.Series(b))
    return float(corr) if pd.notna(corr) else None


def evaluate_threshold(df: pd.DataFrame, prob_col: str, threshold: float) -> dict:
    sub = df[df[prob_col] >= threshold].copy()
    rows = int(len(sub))
    take_rate = float(rows / len(df)) if len(df) else np.nan

    if rows > 0:
        mean_ret = float(sub["target_return_15m"].mean())
        hit_rate = float((sub["target_return_15m"] > 0).mean())
        profitable_after_cost = float((sub["target_return_15m"] > COST_BUFFER).mean())
        robust_score = float(mean_ret * np.sqrt(rows)) if rows >= MIN_ROWS_AT_THRESHOLD else np.nan
    else:
        mean_ret = np.nan
        hit_rate = np.nan
        profitable_after_cost = np.nan
        robust_score = np.nan

    return {
        "rows": rows,
        "take_rate": take_rate,
        "mean_target_return_15m": mean_ret,
        "hit_rate": hit_rate,
        "profitable_after_cost_rate": profitable_after_cost,
        "robust_score": robust_score,
    }


# =========================================================
# LOAD
# =========================================================
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)

require_columns(
    df,
    [
        "timestamp",
        "target_return_15m",
        "pred_return_15m",
        "take_trade",
        "vol_30",
        "trend_ok",
        "breakout_ok",
    ],
    "v2_return_model_scored",
)

df["timestamp"] = parse_timestamp_series(df["timestamp"], TIMEZONE)
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# focus the meta model on rows where base model wasn't strongly bearish
df = df[df["pred_return_15m"] >= BASE_PRED_MIN].copy()

if len(df) < 1000:
    raise ValueError(f"Not enough rows after BASE_PRED_MIN filter: {len(df)}")


# =========================================================
# META FEATURES
# =========================================================
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day_of_week"] = df["timestamp"].dt.dayofweek

df["pred_abs_15m"] = df["pred_return_15m"].abs()
df["pred_x_vol30"] = df["pred_return_15m"] * df["vol_30"]
df["pred_over_vol30"] = df["pred_return_15m"] / (df["vol_30"] + 1e-12)

if "dist_high_15" in df.columns:
    df["pred_x_dist_high_15"] = df["pred_return_15m"] * df["dist_high_15"]

if "dist_sma_15" in df.columns:
    df["pred_x_dist_sma_15"] = df["pred_return_15m"] * df["dist_sma_15"]

if "minutes_to_close" in df.columns:
    df["pred_x_minutes_to_close"] = df["pred_return_15m"] * df["minutes_to_close"]

# meta target: only trades that are profitable after a small cost buffer
df["target_take_trade_meta"] = (df["target_return_15m"] > COST_BUFFER).astype(int)

feature_candidates = [
    "pred_return_15m",
    "pred_abs_15m",
    "pred_x_vol30",
    "pred_over_vol30",
    "vol_30",
    "trend_ok",
    "breakout_ok",
    "hour",
    "minute",
    "day_of_week",
    "minutes_from_open",
    "minutes_to_close",
    "dist_high_15",
    "dist_low_15",
    "dist_sma_15",
    "dist_sma_30",
    "rsi_14",
    "ret_1",
    "ret_5",
    "ret_15",
    "slope_sma_5",
    "slope_sma_15",
    "pos_in_range_15",
    "vol_60",
    "pred_x_dist_high_15",
    "pred_x_dist_sma_15",
    "pred_x_minutes_to_close",
]

feature_cols = [c for c in feature_candidates if c in df.columns and not df[c].isna().all()]

df = df.dropna(subset=feature_cols + ["target_take_trade_meta", "target_return_15m"]).copy()

if len(df) < 1000:
    raise ValueError(f"Not enough rows after meta preprocessing: {len(df)}")

print(f"Rows after preprocessing: {len(df)}")
print(f"Using {len(feature_cols)} meta features")
print(feature_cols)


# =========================================================
# TIME SPLIT
# =========================================================
n = len(df)
test_start = int(n * (1 - TEST_SIZE_FRACTION))
val_start = int(test_start * (1 - VALIDATION_SIZE_FRACTION))

train_df = df.iloc[:val_start].copy()
val_df = df.iloc[val_start:test_start].copy()
test_df = df.iloc[test_start:].copy()

X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
X_test = test_df[feature_cols]

y_train = train_df["target_take_trade_meta"]
y_val = val_df["target_take_trade_meta"]
y_test = test_df["target_take_trade_meta"]

if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
    raise ValueError("Empty train/val/test split.")


# =========================================================
# SAMPLE WEIGHTS
# =========================================================
train_weights = np.clip(train_df["target_return_15m"].abs() / 0.001, 1.0, 10.0)


# =========================================================
# MODEL
# =========================================================
pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=400,
                max_depth=5,
                min_samples_leaf=30,
                l2_regularization=2.0,
                random_state=42,
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train, model__sample_weight=train_weights)

val_prob = pipeline.predict_proba(X_val)[:, 1]
test_prob = pipeline.predict_proba(X_test)[:, 1]

val_pred = (val_prob >= 0.5).astype(int)
test_pred = (test_prob >= 0.5).astype(int)

print("val_prob mean/std:", float(np.mean(val_prob)), float(np.std(val_prob)))
print("test_prob mean/std:", float(np.mean(test_prob)), float(np.std(test_prob)))
print("val_prob min/max:", float(np.min(val_prob)), float(np.max(val_prob)))
print("test_prob min/max:", float(np.min(test_prob)), float(np.max(test_prob)))


# =========================================================
# THRESHOLD SEARCH
# =========================================================
val_eval = val_df.copy()
val_eval["meta_prob_take_trade"] = val_prob

candidate_thresholds = np.unique(
    np.round(np.quantile(val_prob, META_THRESHOLD_QUANTILES), 6)
)

threshold_rows: list[dict] = []
for threshold in candidate_thresholds:
    result = evaluate_threshold(val_eval, "meta_prob_take_trade", float(threshold))
    threshold_rows.append(
        {
            "threshold": float(threshold),
            **result,
        }
    )

threshold_df = pd.DataFrame(threshold_rows).sort_values("threshold").reset_index(drop=True)
valid = threshold_df.dropna(subset=["robust_score"]).copy()

if len(valid) == 0:
    raise ValueError("No valid thresholds found in validation set.")

best_threshold = float(
    valid.sort_values(
        ["robust_score", "mean_target_return_15m", "profitable_after_cost_rate", "rows"],
        ascending=[False, False, False, False],
    ).iloc[0]["threshold"]
)


# =========================================================
# TEST EVALUATION
# =========================================================
test_eval = test_df.copy()
test_eval["meta_prob_take_trade"] = test_prob
test_eval["take_trade_meta"] = (test_eval["meta_prob_take_trade"] >= best_threshold).astype(int)

test_result = evaluate_threshold(test_eval, "meta_prob_take_trade", best_threshold)
test_taken = test_eval[test_eval["take_trade_meta"] == 1].copy()

metrics = {
    "train_rows": int(len(train_df)),
    "val_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "base_pred_min_filter": float(BASE_PRED_MIN),
    "cost_buffer": float(COST_BUFFER),
    "positive_rate_train": float(y_train.mean()),
    "positive_rate_val": float(y_val.mean()),
    "positive_rate_test": float(y_test.mean()),
    "val_accuracy_at_0_5": float(accuracy_score(y_val, val_pred)),
    "test_accuracy_at_0_5": float(accuracy_score(y_test, test_pred)),
    "val_precision_at_0_5": float(precision_score(y_val, val_pred, zero_division=0)),
    "test_precision_at_0_5": float(precision_score(y_test, test_pred, zero_division=0)),
    "val_recall_at_0_5": float(recall_score(y_val, val_pred, zero_division=0)),
    "test_recall_at_0_5": float(recall_score(y_test, test_pred, zero_division=0)),
    "val_roc_auc": float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else None,
    "test_roc_auc": float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else None,
    "best_threshold_from_val": float(best_threshold),
    "test_take_rate": float(test_result["take_rate"]),
    "test_rows_taken": int(test_result["rows"]),
    "test_mean_return_taken": float(test_result["mean_target_return_15m"]) if pd.notna(test_result["mean_target_return_15m"]) else np.nan,
    "test_hit_rate_taken": float(test_result["hit_rate"]) if pd.notna(test_result["hit_rate"]) else np.nan,
    "test_profitable_after_cost_rate": float(test_result["profitable_after_cost_rate"]) if pd.notna(test_result["profitable_after_cost_rate"]) else np.nan,
    "test_mean_meta_prob_taken": float(test_taken["meta_prob_take_trade"].mean()) if len(test_taken) else np.nan,
    "base_model_test_mean_return_all": float(test_df["target_return_15m"].mean()),
    "base_model_test_mean_return_taken": float(test_df.loc[test_df["take_trade"] == 1, "target_return_15m"].mean()) if "take_trade" in test_df.columns and (test_df["take_trade"] == 1).any() else np.nan,
    "test_classification_report_at_0_5": classification_report(y_test, test_pred, zero_division=0),
    "test_pred_vs_actual_corr_on_taken": safe_corr(
        test_taken["meta_prob_take_trade"], test_taken["target_return_15m"]
    ) if len(test_taken) > 1 else None,
}


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
perm = permutation_importance(
    pipeline,
    X_val,
    y_val,
    n_repeats=5,
    random_state=42,
    scoring="roc_auc",
)

feature_importance_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }
).sort_values("importance_mean", ascending=False)

metrics["top_features"] = feature_importance_df.head(15).to_dict(orient="records")
metrics["validation_thresholds"] = threshold_rows


# =========================================================
# SCORE FULL DATA
# =========================================================
full_prob = pipeline.predict_proba(df[feature_cols])[:, 1]
df["meta_prob_take_trade"] = full_prob
df["take_trade_meta"] = (df["meta_prob_take_trade"] >= best_threshold).astype(int)
df["realized_pnl_if_meta_taken"] = np.where(df["take_trade_meta"] == 1, df["target_return_15m"], 0.0)

save_cols = [
    "timestamp",
    "target_return_15m",
    "pred_return_15m",
    "take_trade",
    "meta_prob_take_trade",
    "take_trade_meta",
    "realized_pnl_if_meta_taken",
    *feature_cols,
]
save_cols = [c for c in save_cols if c in df.columns]
scored_df = df[save_cols].copy()


# =========================================================
# SAVE
# =========================================================
bundle = {
    "pipeline": pipeline,
    "feature_cols": feature_cols,
    "target_col": "target_take_trade_meta",
    "best_threshold": best_threshold,
    "base_pred_min_filter": BASE_PRED_MIN,
    "cost_buffer": COST_BUFFER,
    "timezone": TIMEZONE,
}

joblib.dump(bundle, MODEL_PATH)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

scored_df.to_csv(SCORED_PATH, index=False)

print(f"Saved model to: {MODEL_PATH}")
print(f"Saved metrics to: {METRICS_PATH}")
print(f"Saved scored data to: {SCORED_PATH}")

print("\n=== CORE METRICS ===")
for key in [
    "train_rows",
    "val_rows",
    "test_rows",
    "positive_rate_train",
    "positive_rate_val",
    "positive_rate_test",
    "val_roc_auc",
    "test_roc_auc",
    "best_threshold_from_val",
    "test_take_rate",
    "test_rows_taken",
    "test_mean_return_taken",
    "test_hit_rate_taken",
    "test_profitable_after_cost_rate",
]:
    print(f"{key}: {metrics[key]}")

print("\n=== TOP FEATURES ===")
print(feature_importance_df.head(15).to_string(index=False))

print("\n=== VALIDATION THRESHOLDS ===")
print(threshold_df.to_string(index=False))