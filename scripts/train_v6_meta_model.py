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

INPUT_PATH = PROJECT_ROOT / "outputs" / "training" / "labeled_predictions.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v6_meta_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "v6_meta_model.joblib"
METRICS_PATH = OUTPUT_DIR / "v6_meta_model_metrics.json"
SCORED_PATH = OUTPUT_DIR / "v6_meta_model_scored.csv"


# =========================================================
# CONFIG
# =========================================================
TIMEZONE = "America/New_York"

TEST_SIZE_FRACTION = 0.20
VALIDATION_SIZE_FRACTION = 0.15

BASE_PRED_MIN_ABS = 0.0
MIN_ROWS_AT_THRESHOLD = 50

# Profit threshold for "good trade"
COST_BUFFER = 0.0001
TARGET_PROFIT_THRESHOLD = 0.0005

PROBABILITY_QUANTILES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95]
TOP_K_QUANTILE_DEFAULT = 0.80


# =========================================================
# HELPERS
# =========================================================
def require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def safe_corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float | None:
    corr = pd.Series(a).corr(pd.Series(b))
    return float(corr) if pd.notna(corr) else None


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    ts = ts.dt.tz_convert(TIMEZONE)

    df["timestamp"] = ts
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek

    minutes = df["hour"] * 60 + df["minute"]
    df["minutes_from_open"] = minutes - (9 * 60 + 30)
    df["minutes_to_close"] = (16 * 60) - minutes

    df["is_opening_window"] = (df["minutes_from_open"] <= 30).astype(int)
    df["is_midday"] = ((df["minutes_from_open"] > 90) & (df["minutes_to_close"] > 120)).astype(int)
    df["is_power_hour"] = (df["minutes_to_close"] <= 60).astype(int)

    return df


def normalize_signal_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "signal" not in df.columns:
        df["signal"] = "NO_TRADE"

    signal_map = {
        "NO_TRADE": 0,
        "LONG": 1,
        "SHORT": -1,
        "LONG_BIAS": 1,
        "SHORT_BIAS": -1,
        "LONG_BREAKOUT_CONTINUATION": 2,
        "SHORT_BREAKOUT_CONTINUATION": -2,
        "LONG_BREAKOUT_REVERSAL_RISK": 1,
        "SHORT_BREAKOUT_REVERSAL_RISK": -1,
    }
    df["signal_num"] = df["signal"].map(signal_map).fillna(0)
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    require_columns(df, ["actual_return_15m"], "labeled_predictions.csv")

    df["target_class"] = (df["actual_return_15m"] > TARGET_PROFIT_THRESHOLD).astype(int)
    df["target_positive"] = (df["actual_return_15m"] > 0).astype(int)
    df["target_after_cost"] = (df["actual_return_15m"] > COST_BUFFER).astype(int)

    return df


def build_meta_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    # Base prediction structure
    df["pred_abs_5m"] = df["pred_return_5m"].abs()
    df["pred_abs_15m"] = df["pred_return_15m"].abs()
    df["pred_abs_30m"] = df["pred_return_30m"].abs()

    df["pred_spread_5_15"] = df["pred_return_15m"] - df["pred_return_5m"]
    df["pred_spread_15_30"] = df["pred_return_30m"] - df["pred_return_15m"]
    df["pred_slope_5_15_30"] = df["pred_return_30m"] - df["pred_return_5m"]

    df["pred_sign_5m"] = np.sign(df["pred_return_5m"])
    df["pred_sign_15m"] = np.sign(df["pred_return_15m"])
    df["pred_sign_30m"] = np.sign(df["pred_return_30m"])

    df["confidence_x_pred_15m"] = df["confidence"] * df["pred_return_15m"]
    df["confidence_x_abs_pred_15m"] = df["confidence"] * df["pred_abs_15m"]

    df["pred_rank_proxy"] = (
        0.20 * df["pred_return_5m"]
        + 0.60 * df["pred_return_15m"]
        + 0.20 * df["pred_return_30m"]
    )

    df["pred_consensus"] = (
        (np.sign(df["pred_return_5m"]) == np.sign(df["pred_return_15m"]))
        & (np.sign(df["pred_return_15m"]) == np.sign(df["pred_return_30m"]))
    ).astype(int)

    df["pred_curve_strength"] = (
        df["pred_abs_5m"] + df["pred_abs_15m"] + df["pred_abs_30m"]
    ) / 3.0

    # Signal quality
    df["confidence_rank"] = df["confidence"].rank(pct=True)
    df["pred_return_rank"] = df["pred_return_15m"].rank(pct=True)
    df["pred_abs_rank"] = df["pred_abs_15m"].rank(pct=True)

    feature_cols = [
        "pred_return_5m",
        "pred_return_15m",
        "pred_return_30m",
        "pred_abs_5m",
        "pred_abs_15m",
        "pred_abs_30m",
        "pred_spread_5_15",
        "pred_spread_15_30",
        "pred_slope_5_15_30",
        "pred_sign_5m",
        "pred_sign_15m",
        "pred_sign_30m",
        "confidence",
        "confidence_x_pred_15m",
        "confidence_x_abs_pred_15m",
        "pred_rank_proxy",
        "pred_consensus",
        "pred_curve_strength",
        "confidence_rank",
        "pred_return_rank",
        "pred_abs_rank",
        "signal_num",
        "hour",
        "minute",
        "day_of_week",
        "minutes_from_open",
        "minutes_to_close",
        "is_opening_window",
        "is_midday",
        "is_power_hour",
    ]

    optional_cols = [
        "breakout_up_prob_15m",
        "breakout_down_prob_15m",
        "continuation_prob_15m",
        "breakout_ok",
        "trend_ok",
        "vol_30",
        "vol_60",
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
        "regime_active",
        "regime_confidence",
        "trading_enabled",
        "selected_universe_num",
        "quantum_score",
        "quantum_energy",
        "quantum_dispersion",
        "quantum_alignment",
    ]
    feature_cols.extend([c for c in optional_cols if c in df.columns])

    if "vol_30" in df.columns:
        df["pred_over_vol30"] = df["pred_return_15m"] / (df["vol_30"].abs() + 1e-9)
        df["pred_x_vol30"] = df["pred_return_15m"] * df["vol_30"]
        feature_cols += ["pred_over_vol30", "pred_x_vol30"]

    if "vol_60" in df.columns:
        df["pred_over_vol60"] = df["pred_return_15m"] / (df["vol_60"].abs() + 1e-9)
        feature_cols += ["pred_over_vol60"]

    if "dist_high_15" in df.columns:
        df["pred_x_dist_high_15"] = df["pred_return_15m"] * df["dist_high_15"]
        feature_cols += ["pred_x_dist_high_15"]

    if "minutes_to_close" in df.columns:
        df["pred_x_minutes_to_close"] = df["pred_return_15m"] * df["minutes_to_close"]
        feature_cols += ["pred_x_minutes_to_close"]

    if "regime_active" in df.columns:
        df["pred_x_regime"] = df["pred_return_15m"] * df["regime_active"]
        feature_cols += ["pred_x_regime"]

    feature_cols = [c for c in feature_cols if c in df.columns and not df[c].isna().all()]
    df = df.dropna(subset=feature_cols + ["target_class", "actual_return_15m"]).copy()

    return df, feature_cols


def evaluate_threshold(df: pd.DataFrame, prob_col: str, threshold: float) -> dict:
    sub = df[df[prob_col] >= threshold].copy()
    rows = int(len(sub))
    take_rate = float(rows / len(df)) if len(df) else np.nan

    if rows > 0:
        mean_ret = float(sub["actual_return_15m"].mean())
        hit_rate = float((sub["actual_return_15m"] > 0).mean())
        profitable_after_cost = float((sub["actual_return_15m"] > COST_BUFFER).mean())
        class_hit_rate = float((sub["target_class"] == 1).mean())
        robust_score = float(mean_ret * np.sqrt(rows)) if rows >= MIN_ROWS_AT_THRESHOLD else np.nan
    else:
        mean_ret = np.nan
        hit_rate = np.nan
        profitable_after_cost = np.nan
        class_hit_rate = np.nan
        robust_score = np.nan

    return {
        "rows": rows,
        "take_rate": take_rate,
        "mean_target_return_15m": mean_ret,
        "hit_rate": hit_rate,
        "profitable_after_cost_rate": profitable_after_cost,
        "target_class_rate": class_hit_rate,
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
        "pred_return_5m",
        "pred_return_15m",
        "pred_return_30m",
        "confidence",
        "signal",
        "actual_return_15m",
    ],
    "labeled_predictions.csv",
)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
df = df.dropna(subset=["timestamp"]).copy()

print(f"Loaded labeled dataset with {len(df)} rows")
print(df.head())
print("timestamp min/max:", df["timestamp"].min(), df["timestamp"].max())


# =========================================================
# FEATURE BUILD
# =========================================================
df = build_time_features(df)
df = normalize_signal_column(df)
df = build_targets(df)

df = df[df["pred_return_15m"].abs() >= BASE_PRED_MIN_ABS].copy()

df, feature_cols = build_meta_features(df)

if len(df) < 1000:
    raise ValueError(f"Not enough rows after preprocessing: {len(df)}")

print(f"Rows after preprocessing: {len(df)}")
print(f"Using {len(feature_cols)} meta features")
print(feature_cols)


# =========================================================
# TIME SPLIT
# =========================================================
if df.empty:
    raise ValueError("Dataset is empty after preprocessing.")

df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)
if n < 3:
    raise ValueError(f"Need at least 3 rows for train/val/test split, got {n}")

test_size = max(1, int(n * TEST_SIZE_FRACTION))
remaining = n - test_size
val_size = max(1, int(remaining * VALIDATION_SIZE_FRACTION))
train_size = n - test_size - val_size

if train_size < 1:
    raise ValueError(
        f"Not enough rows after split sizing: n={n}, "
        f"train_size={train_size}, val_size={val_size}, test_size={test_size}"
    )

train_df = df.iloc[:train_size].copy()
val_df = df.iloc[train_size:train_size + val_size].copy()
test_df = df.iloc[train_size + val_size:].copy()

print(f"Total rows: {n}")
print(f"Train rows: {len(train_df)}")
print(f"Val rows:   {len(val_df)}")
print(f"Test rows:  {len(test_df)}")

X_train = train_df[feature_cols].copy()
X_val = val_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

y_train = train_df["target_class"].copy()
y_val = val_df["target_class"].copy()
y_test = test_df["target_class"].copy()


# =========================================================
# MODEL
# =========================================================
sample_weight = np.where(train_df["target_class"] == 1, 2.0, 1.0)
sample_weight = sample_weight * np.clip(train_df["actual_return_15m"].abs() / 0.001, 1.0, 5.0)

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

pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)

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
val_eval["meta_prob"] = val_prob

candidate_thresholds = np.unique(
    np.round(np.quantile(val_prob, PROBABILITY_QUANTILES), 8)
)

threshold_rows: list[dict] = []
for threshold in candidate_thresholds:
    result = evaluate_threshold(
        df=val_eval,
        prob_col="meta_prob",
        threshold=float(threshold),
    )
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
        ["robust_score", "mean_target_return_15m", "target_class_rate", "rows"],
        ascending=[False, False, False, False],
    ).iloc[0]["threshold"]
)

topk_threshold = float(np.quantile(val_prob, TOP_K_QUANTILE_DEFAULT))


# =========================================================
# TEST EVALUATION
# =========================================================
test_eval = test_df.copy()
test_eval["meta_prob"] = test_prob
test_eval["take_trade_meta"] = (test_eval["meta_prob"] >= best_threshold).astype(int)

test_result = evaluate_threshold(
    df=test_eval,
    prob_col="meta_prob",
    threshold=best_threshold,
)

test_taken = test_eval[test_eval["take_trade_meta"] == 1].copy()

metrics = {
    "train_rows": int(len(train_df)),
    "val_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "base_pred_min_abs_filter": float(BASE_PRED_MIN_ABS),
    "cost_buffer": float(COST_BUFFER),
    "target_profit_threshold": float(TARGET_PROFIT_THRESHOLD),
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
    "topk_threshold_from_val": float(topk_threshold),
    "test_take_rate": float(test_result["take_rate"]),
    "test_rows_taken": int(test_result["rows"]),
    "test_mean_return_taken": float(test_result["mean_target_return_15m"]) if pd.notna(test_result["mean_target_return_15m"]) else np.nan,
    "test_hit_rate_taken": float(test_result["hit_rate"]) if pd.notna(test_result["hit_rate"]) else np.nan,
    "test_profitable_after_cost_rate": float(test_result["profitable_after_cost_rate"]) if pd.notna(test_result["profitable_after_cost_rate"]) else np.nan,
    "test_target_class_rate_taken": float(test_result["target_class_rate"]) if pd.notna(test_result["target_class_rate"]) else np.nan,
    "test_mean_meta_prob_taken": float(test_taken["meta_prob"].mean()) if len(test_taken) else np.nan,
    "test_prob_vs_actual_corr_on_taken": safe_corr(
        test_taken["meta_prob"], test_taken["actual_return_15m"]
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

metrics["top_features"] = feature_importance_df.head(20).to_dict(orient="records")
metrics["validation_thresholds"] = threshold_rows


# =========================================================
# SCORE FULL DATA
# =========================================================
full_prob = pipeline.predict_proba(df[feature_cols])[:, 1]
df["meta_prob"] = full_prob
df["take_trade_meta"] = (df["meta_prob"] >= best_threshold).astype(int)
df["realized_pnl_if_meta_taken"] = np.where(df["take_trade_meta"] == 1, df["actual_return_15m"], 0.0)

save_cols = [
    "timestamp",
    "actual_return_15m",
    "target_class",
    "pred_return_5m",
    "pred_return_15m",
    "pred_return_30m",
    "confidence",
    "signal",
    "meta_prob",
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
    "target_col": "target_class",
    "best_threshold": best_threshold,
    "topk_threshold": topk_threshold,
    "base_pred_min_abs_filter": BASE_PRED_MIN_ABS,
    "cost_buffer": COST_BUFFER,
    "target_profit_threshold": TARGET_PROFIT_THRESHOLD,
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
    "test_target_class_rate_taken",
]:
    print(f"{key}: {metrics[key]}")

print("\n=== TOP FEATURES ===")
print(feature_importance_df.head(20).to_string(index=False))

print("\n=== VALIDATION THRESHOLDS ===")
print(threshold_df.to_string(index=False))