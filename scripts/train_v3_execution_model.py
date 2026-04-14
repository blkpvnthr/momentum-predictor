from __future__ import annotations

import json
from itertools import product
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

INPUT_PATH = PROJECT_ROOT / "data" / "market_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v3_execution_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "v3_execution_model.joblib"
METRICS_PATH = OUTPUT_DIR / "v3_execution_model_metrics.json"
SCORED_PATH = OUTPUT_DIR / "v3_execution_model_scored.csv"


# =========================================================
# CONFIG
# =========================================================
TIMEZONE = "America/New_York"
TARGET_HORIZON_MINUTES = 15

# Strong target: only meaningful 15m wins count
COST_BUFFER = 0.0015

TEST_SIZE_FRACTION = 0.20
VALIDATION_SIZE_FRACTION = 0.15

MINUTES_FROM_OPEN = 10
MINUTES_TO_CLOSE_MIN = TARGET_HORIZON_MINUTES + 10

MIN_ROWS_VALIDATION = 50
MIN_ROWS_TEST = 50


# =========================================================
# HELPERS
# =========================================================
def require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}

    if "timestamp" not in df.columns:
        if "time" in df.columns:
            rename_map["time"] = "timestamp"
        elif "datetime" in df.columns:
            rename_map["datetime"] = "timestamp"
        elif "Datetime" in df.columns:
            rename_map["Datetime"] = "timestamp"

    if "close" not in df.columns:
        if "c" in df.columns:
            rename_map["c"] = "close"
        elif "Close" in df.columns:
            rename_map["Close"] = "close"
        elif "close_price" in df.columns:
            rename_map["close_price"] = "close"

    if rename_map:
        df = df.rename(columns=rename_map)

    require_columns(df, ["timestamp", "close"], "market_data")
    return df


def parse_timestamp_series(series: pd.Series, timezone: str) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_convert(timezone)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    minutes = ts.dt.hour * 60 + ts.dt.minute

    df["session_date"] = ts.dt.date
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek

    df["minutes_from_open"] = minutes - (9 * 60 + 30)
    df["minutes_to_close"] = (16 * 60) - minutes

    df["is_opening_window"] = (df["minutes_from_open"] <= 30).astype(int)
    df["is_midday"] = ((df["minutes_from_open"] > 90) & (df["minutes_to_close"] > 120)).astype(int)
    df["is_power_hour"] = (df["minutes_to_close"] <= 60).astype(int)

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    # Returns
    df["ret_1"] = close.pct_change(1)
    df["ret_2"] = close.pct_change(2)
    df["ret_3"] = close.pct_change(3)
    df["ret_4"] = close.pct_change(4)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_15"] = close.pct_change(15)
    df["ret_30"] = close.pct_change(30)
    df["ret_60"] = close.pct_change(60)

    # Momentum
    df["mom_3"] = close / close.shift(3) - 1.0
    df["mom_5"] = close / close.shift(5) - 1.0
    df["mom_15"] = close / close.shift(15) - 1.0
    df["mom_30"] = close / close.shift(30) - 1.0
    df["mom_60"] = close / close.shift(60) - 1.0

    # Volatility
    df["vol_5"] = df["ret_1"].rolling(5, min_periods=5).std()
    df["vol_15"] = df["ret_1"].rolling(15, min_periods=15).std()
    df["vol_30"] = df["ret_1"].rolling(30, min_periods=30).std()
    df["vol_60"] = df["ret_1"].rolling(60, min_periods=60).std()

    # Trend context
    df["sma_5"] = close.rolling(5, min_periods=5).mean()
    df["sma_15"] = close.rolling(15, min_periods=15).mean()
    df["sma_30"] = close.rolling(30, min_periods=30).mean()
    df["sma_60"] = close.rolling(60, min_periods=60).mean()

    df["dist_sma_5"] = close / df["sma_5"] - 1.0
    df["dist_sma_15"] = close / df["sma_15"] - 1.0
    df["dist_sma_30"] = close / df["sma_30"] - 1.0
    df["dist_sma_60"] = close / df["sma_60"] - 1.0

    df["slope_sma_5"] = df["sma_5"].pct_change(3)
    df["slope_sma_15"] = df["sma_15"].pct_change(3)
    df["slope_sma_30"] = df["sma_30"].pct_change(3)
    df["slope_sma_60"] = df["sma_60"].pct_change(5)

    # Range / breakout features
    df["rolling_high_15"] = close.rolling(15, min_periods=15).max()
    df["rolling_low_15"] = close.rolling(15, min_periods=15).min()
    df["rolling_high_30"] = close.rolling(30, min_periods=30).max()
    df["rolling_low_30"] = close.rolling(30, min_periods=30).min()

    df["pos_in_range_15"] = (close - df["rolling_low_15"]) / (
        (df["rolling_high_15"] - df["rolling_low_15"]) + 1e-12
    )
    df["pos_in_range_30"] = (close - df["rolling_low_30"]) / (
        (df["rolling_high_30"] - df["rolling_low_30"]) + 1e-12
    )

    df["dist_high_15"] = close / (df["rolling_high_15"] + 1e-12) - 1.0
    df["dist_low_15"] = close / (df["rolling_low_15"] + 1e-12) - 1.0
    df["dist_high_30"] = close / (df["rolling_high_30"] + 1e-12) - 1.0
    df["dist_low_30"] = close / (df["rolling_low_30"] + 1e-12) - 1.0

    # Regime-ish features
    df["accel_1_5"] = df["ret_1"] - df["ret_5"]
    df["accel_3_15"] = df["ret_3"] - df["ret_15"]
    df["vol_ratio_5_30"] = df["vol_5"] / (df["vol_30"] + 1e-12)
    df["vol_ratio_15_60"] = df["vol_15"] / (df["vol_60"] + 1e-12)

    df["rsi_14"] = compute_rsi(close, period=14)

    # Rule-layer helper columns
    df["trend_ok"] = (
        (df["ret_5"] > 0) &
        (df["slope_sma_15"] > 0) &
        (df["dist_sma_15"] > -0.002)
    ).astype(int)

    df["breakout_ok"] = (
        (df["pos_in_range_15"] >= 0.70) &
        (df["dist_high_15"] >= -0.0025)
    ).astype(int)

    return df


def add_targets(df: pd.DataFrame, horizon: int, cost_buffer: float) -> pd.DataFrame:
    future_close = df["close"].shift(-horizon)
    df["target_return_15m"] = future_close / df["close"] - 1.0
    df["target_long_profit_15m"] = (df["target_return_15m"] > cost_buffer).astype(int)
    return df


def build_feature_list(df: pd.DataFrame) -> list[str]:
    candidate_feature_cols = [
        "ret_1", "ret_2", "ret_3", "ret_4", "ret_5", "ret_10", "ret_15", "ret_30", "ret_60",
        "mom_3", "mom_5", "mom_15", "mom_30", "mom_60",
        "vol_5", "vol_15", "vol_30", "vol_60",
        "dist_sma_5", "dist_sma_15", "dist_sma_30", "dist_sma_60",
        "slope_sma_5", "slope_sma_15", "slope_sma_30", "slope_sma_60",
        "pos_in_range_15", "pos_in_range_30",
        "dist_high_15", "dist_low_15", "dist_high_30", "dist_low_30",
        "accel_1_5", "accel_3_15",
        "vol_ratio_5_30", "vol_ratio_15_60",
        "rsi_14",
        "hour", "minute", "day_of_week",
        "minutes_from_open", "minutes_to_close",
        "is_opening_window", "is_midday", "is_power_hour",
    ]
    return [c for c in candidate_feature_cols if c in df.columns and not df[c].isna().all()]


def evaluate_execution_rules(
    df: pd.DataFrame,
    prob_col: str,
    cost_buffer: float,
    prob_threshold: float,
    vol_quantile: float,
    require_trend: bool,
    require_breakout: bool,
) -> dict:
    vol_cut = df["vol_30"].quantile(vol_quantile)

    mask = df[prob_col] >= prob_threshold
    mask &= df["vol_30"] >= vol_cut

    if require_trend:
        mask &= df["trend_ok"] == 1

    if require_breakout:
        mask &= df["breakout_ok"] == 1

    sub = df.loc[mask].copy()
    rows = len(sub)

    if rows == 0:
        return {
            "rows": 0,
            "take_rate": 0.0,
            "precision_like": np.nan,
            "mean_target_return_15m": np.nan,
            "robust_score": np.nan,
        }

    precision_like = float((sub["target_return_15m"] > cost_buffer).mean())
    mean_ret = float(sub["target_return_15m"].mean())
    take_rate = float(rows / len(df))
    robust_score = float(mean_ret * np.sqrt(rows))

    return {
        "rows": int(rows),
        "take_rate": take_rate,
        "precision_like": precision_like,
        "mean_target_return_15m": mean_ret,
        "robust_score": robust_score,
    }


# =========================================================
# LOAD + FEATURES
# =========================================================
df = pd.read_csv(INPUT_PATH)
df = normalize_columns(df)

df["timestamp"] = parse_timestamp_series(df["timestamp"], TIMEZONE)
df = df.dropna(subset=["timestamp", "close"]).copy()
df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

df = add_time_features(df)
df = add_price_features(df)
df = add_targets(df, TARGET_HORIZON_MINUTES, COST_BUFFER)

df = df[df["minutes_from_open"] >= MINUTES_FROM_OPEN].copy()
df = df[df["minutes_to_close"] >= MINUTES_TO_CLOSE_MIN].copy()
df = df.dropna(subset=["target_return_15m"]).copy()

feature_cols = build_feature_list(df)
target_col = "target_long_profit_15m"

df = df.dropna(subset=[target_col, "target_return_15m"]).copy()
df = df.dropna(subset=feature_cols).copy()

if len(df) < 1000:
    raise ValueError(f"Not enough rows after feature engineering: {len(df)}")

print(f"Using {len(feature_cols)} features")
print(feature_cols)


# =========================================================
# TIME SPLIT
# =========================================================
df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)
test_start = int(n * (1 - TEST_SIZE_FRACTION))
val_start = int(test_start * (1 - VALIDATION_SIZE_FRACTION))

train_df = df.iloc[:val_start].copy()
val_df = df.iloc[val_start:test_start].copy()
test_df = df.iloc[test_start:].copy()

if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
    raise ValueError("One or more train/val/test partitions are empty.")

X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
X_test = test_df[feature_cols]

y_train = train_df[target_col]
y_val = val_df[target_col]
y_test = test_df[target_col]


# =========================================================
# MODEL
# =========================================================
train_weights = np.clip(train_df["target_return_15m"].abs() / 0.001, 1.0, 10.0)

pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=500,
                max_depth=6,
                min_samples_leaf=40,
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

val_eval = val_df.copy()
val_eval["prob_long_profit_15m"] = val_prob

test_eval = test_df.copy()
test_eval["prob_long_profit_15m"] = test_prob


# =========================================================
# EXECUTION RULE SEARCH
# =========================================================
prob_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
vol_quantiles = [0.50, 0.60, 0.70, 0.80]
trend_options = [False, True]
breakout_options = [False, True]

rule_rows: list[dict] = []

for prob_threshold, vol_quantile, require_trend, require_breakout in product(
    prob_thresholds,
    vol_quantiles,
    trend_options,
    breakout_options,
):
    result = evaluate_execution_rules(
        df=val_eval,
        prob_col="prob_long_profit_15m",
        cost_buffer=COST_BUFFER,
        prob_threshold=prob_threshold,
        vol_quantile=vol_quantile,
        require_trend=require_trend,
        require_breakout=require_breakout,
    )

    rule_rows.append(
        {
            "prob_threshold": prob_threshold,
            "vol_quantile": vol_quantile,
            "require_trend": require_trend,
            "require_breakout": require_breakout,
            **result,
        }
    )

rules_df = pd.DataFrame(rule_rows)

eligible_rules = rules_df[
    (rules_df["rows"] >= MIN_ROWS_VALIDATION) &
    (rules_df["mean_target_return_15m"] > 0)
].copy()

if len(eligible_rules) == 0:
    best_rule = rules_df.sort_values(
        ["robust_score", "precision_like", "rows"],
        ascending=[False, False, False],
    ).iloc[0]
else:
    best_rule = eligible_rules.sort_values(
        ["robust_score", "precision_like", "rows"],
        ascending=[False, False, False],
    ).iloc[0]

best_prob_threshold = float(best_rule["prob_threshold"])
best_vol_quantile = float(best_rule["vol_quantile"])
best_require_trend = bool(best_rule["require_trend"])
best_require_breakout = bool(best_rule["require_breakout"])


# =========================================================
# APPLY BEST EXECUTION RULE TO TEST
# =========================================================
test_result = evaluate_execution_rules(
    df=test_eval,
    prob_col="prob_long_profit_15m",
    cost_buffer=COST_BUFFER,
    prob_threshold=best_prob_threshold,
    vol_quantile=best_vol_quantile,
    require_trend=best_require_trend,
    require_breakout=best_require_breakout,
)

test_vol_cut = test_eval["vol_30"].quantile(best_vol_quantile)

test_mask = test_eval["prob_long_profit_15m"] >= best_prob_threshold
test_mask &= test_eval["vol_30"] >= test_vol_cut

if best_require_trend:
    test_mask &= test_eval["trend_ok"] == 1

if best_require_breakout:
    test_mask &= test_eval["breakout_ok"] == 1

test_eval["take_trade"] = test_mask.astype(int)
test_taken = test_eval[test_eval["take_trade"] == 1].copy()


# =========================================================
# METRICS
# =========================================================
metrics = {
    "train_rows": int(len(train_df)),
    "val_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
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
    "best_rule": {
        "prob_threshold": best_prob_threshold,
        "vol_quantile": best_vol_quantile,
        "require_trend": best_require_trend,
        "require_breakout": best_require_breakout,
    },
    "test_take_rate": float(test_result["take_rate"]),
    "test_rows_taken": int(test_result["rows"]),
    "test_hit_rate_taken": float(test_result["precision_like"]) if pd.notna(test_result["precision_like"]) else np.nan,
    "test_mean_return_taken": float(test_result["mean_target_return_15m"]) if pd.notna(test_result["mean_target_return_15m"]) else np.nan,
    "test_robust_score": float(test_result["robust_score"]) if pd.notna(test_result["robust_score"]) else np.nan,
    "test_mean_prob_taken": float(test_taken["prob_long_profit_15m"].mean()) if len(test_taken) else np.nan,
    "test_classification_report_at_0_5": classification_report(y_test, test_pred, zero_division=0),
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
metrics["validation_rule_grid"] = rule_rows


# =========================================================
# SCORE FULL DATASET
# =========================================================
full_prob = pipeline.predict_proba(df[feature_cols])[:, 1]
df["prob_long_profit_15m"] = full_prob

full_vol_cut = df["vol_30"].quantile(best_vol_quantile)

full_mask = df["prob_long_profit_15m"] >= best_prob_threshold
full_mask &= df["vol_30"] >= full_vol_cut

if best_require_trend:
    full_mask &= df["trend_ok"] == 1

if best_require_breakout:
    full_mask &= df["breakout_ok"] == 1

df["take_trade"] = full_mask.astype(int)
df["realized_pnl_if_taken"] = np.where(df["take_trade"] == 1, df["target_return_15m"], 0.0)

scored_cols = [
    "timestamp",
    "close",
    "target_return_15m",
    "target_long_profit_15m",
    "prob_long_profit_15m",
    "trend_ok",
    "breakout_ok",
    "vol_30",
    "take_trade",
    "realized_pnl_if_taken",
    *feature_cols,
]
scored_df = df[scored_cols].copy()


# =========================================================
# SAVE
# =========================================================
bundle = {
    "pipeline": pipeline,
    "feature_cols": feature_cols,
    "target_col": target_col,
    "cost_buffer": COST_BUFFER,
    "best_rule": {
        "prob_threshold": best_prob_threshold,
        "vol_quantile": best_vol_quantile,
        "require_trend": best_require_trend,
        "require_breakout": best_require_breakout,
    },
    "timezone": TIMEZONE,
    "target_horizon_minutes": TARGET_HORIZON_MINUTES,
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
    "test_take_rate",
    "test_rows_taken",
    "test_hit_rate_taken",
    "test_mean_return_taken",
]:
    print(f"{key}: {metrics[key]}")

print("\n=== BEST RULE ===")
print(metrics["best_rule"])

print("\n=== TOP FEATURES ===")
print(feature_importance_df.head(15).to_string(index=False))

print("\n=== VALIDATION RULE GRID (TOP 20) ===")
print(
    rules_df.sort_values(
        ["robust_score", "precision_like", "rows"],
        ascending=[False, False, False],
    ).head(20).to_string(index=False)
)