from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


# =========================================================
# PATHS
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "market_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2_return_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "v2_return_model.joblib"
METRICS_PATH = OUTPUT_DIR / "v2_return_model_metrics.json"
SCORED_PATH = OUTPUT_DIR / "v2_return_model_scored.csv"


# =========================================================
# CONFIG
# =========================================================
TIMEZONE = "America/New_York"
TARGET_HORIZON_MINUTES = 15

TEST_SIZE_FRACTION = 0.20
VALIDATION_SIZE_FRACTION = 0.15

MINUTES_FROM_OPEN = 5
MINUTES_TO_CLOSE_MIN = TARGET_HORIZON_MINUTES + 5

MIN_ROWS_AT_THRESHOLD = 50

# Optional execution filter layer
USE_EXECUTION_FILTERS = True
VOL_FILTER_QUANTILE = 0.80
BREAKOUT_DIST_HIGH_15_MIN = -0.0025
TREND_DIST_SMA_15_MIN = -0.0020
TREND_RET_5_MIN = 0.0
TREND_SLOPE_SMA_15_MIN = 0.0


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
    df["is_midday"] = (
        (df["minutes_from_open"] > 90) & (df["minutes_to_close"] > 120)
    ).astype(int)
    df["is_power_hour"] = (df["minutes_to_close"] <= 60).astype(int)

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    # Returns / momentum
    df["ret_1"] = close.pct_change(1)
    df["ret_2"] = close.pct_change(2)
    df["ret_3"] = close.pct_change(3)
    df["ret_4"] = close.pct_change(4)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_15"] = close.pct_change(15)
    df["ret_30"] = close.pct_change(30)
    df["ret_60"] = close.pct_change(60)

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

    # Trend
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

    # Range position / breakout pressure
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

    # Acceleration / regime-shift style features
    df["accel_1_5"] = df["ret_1"] - df["ret_5"]
    df["accel_3_15"] = df["ret_3"] - df["ret_15"]
    df["vol_ratio_5_30"] = df["vol_5"] / (df["vol_30"] + 1e-12)
    df["vol_ratio_15_60"] = df["vol_15"] / (df["vol_60"] + 1e-12)

    # Oscillator
    df["rsi_14"] = compute_rsi(close, period=14)

    # Execution helpers
    df["trend_ok"] = (
        (df["ret_5"] > TREND_RET_5_MIN)
        & (df["slope_sma_15"] > TREND_SLOPE_SMA_15_MIN)
        & (df["dist_sma_15"] > TREND_DIST_SMA_15_MIN)
    ).astype(int)

    df["breakout_ok"] = (df["dist_high_15"] >= BREAKOUT_DIST_HIGH_15_MIN).astype(int)

    return df


def add_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    future_close = df["close"].shift(-horizon)
    df["target_return_15m"] = future_close / df["close"] - 1.0
    return df


def evaluate_threshold(
    df: pd.DataFrame,
    pred_col: str,
    threshold: float,
    apply_filters: bool,
    vol_filter_quantile: float,
) -> dict:
    working = df.copy()

    mask = working[pred_col] >= threshold

    if apply_filters:
        vol_cut = working["vol_30"].quantile(vol_filter_quantile)
        mask &= working["vol_30"] >= vol_cut
        mask &= working["trend_ok"] == 1
        mask &= working["breakout_ok"] == 1

    sub = working.loc[mask].copy()
    rows = int(len(sub))
    take_rate = float(rows / len(working)) if len(working) else np.nan

    if rows > 0:
        mean_ret = float(sub["target_return_15m"].mean())
        hit_rate = float((sub["target_return_15m"] > 0).mean())
        robust_score = float(mean_ret * np.sqrt(rows)) if rows >= MIN_ROWS_AT_THRESHOLD else np.nan
    else:
        mean_ret = np.nan
        hit_rate = np.nan
        robust_score = np.nan

    return {
        "rows": rows,
        "take_rate": take_rate,
        "mean_target_return_15m": mean_ret,
        "hit_rate": hit_rate,
        "robust_score": robust_score,
    }


# =========================================================
# LOAD + FEATURE BUILD
# =========================================================
df = pd.read_csv(INPUT_PATH)
df = normalize_columns(df)

df["timestamp"] = parse_timestamp_series(df["timestamp"], TIMEZONE)
df = df.dropna(subset=["timestamp", "close"]).copy()
df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

df = add_time_features(df)
df = add_price_features(df)
df = add_targets(df, TARGET_HORIZON_MINUTES)

df = df[df["minutes_from_open"] >= MINUTES_FROM_OPEN].copy()
df = df[df["minutes_to_close"] >= MINUTES_TO_CLOSE_MIN].copy()
df = df.dropna(subset=["target_return_15m"]).copy()

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

feature_cols = [
    col for col in candidate_feature_cols
    if col in df.columns and not df[col].isna().all()
]

target_col = "target_return_15m"

df = df.dropna(subset=[target_col]).copy()
df = df.dropna(subset=feature_cols).copy()

if len(df) < 1000:
    raise ValueError(f"Not enough rows after feature engineering: {len(df)}")

print(f"Using {len(feature_cols)} features")
print(feature_cols)


# =========================================================
# TIME-BASED SPLIT
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
            HistGradientBoostingRegressor(
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

val_pred = pipeline.predict(X_val)
test_pred = pipeline.predict(X_test)

print("val_pred mean/std:", float(np.mean(val_pred)), float(np.std(val_pred)))
print("test_pred mean/std:", float(np.mean(test_pred)), float(np.std(test_pred)))
print("val_pred min/max:", float(np.min(val_pred)), float(np.max(val_pred)))
print("test_pred min/max:", float(np.min(test_pred)), float(np.max(test_pred)))


# =========================================================
# THRESHOLD SEARCH
# Search the upper tail of actual predicted values
# =========================================================
threshold_rows: list[dict] = []

val_eval = val_df.copy()
val_eval["pred_return_15m"] = val_pred

candidate_thresholds = np.unique(
    np.round(
        np.quantile(
            val_pred,
            [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99],
        ),
        6,
    )
)

for threshold in candidate_thresholds:
    result = evaluate_threshold(
        df=val_eval,
        pred_col="pred_return_15m",
        threshold=float(threshold),
        apply_filters=USE_EXECUTION_FILTERS,
        vol_filter_quantile=VOL_FILTER_QUANTILE,
    )

    threshold_rows.append(
        {
            "threshold": float(threshold),
            "filters_enabled": USE_EXECUTION_FILTERS,
            "vol_filter_quantile": VOL_FILTER_QUANTILE if USE_EXECUTION_FILTERS else np.nan,
            **result,
        }
    )

threshold_df = pd.DataFrame(threshold_rows).sort_values("threshold").reset_index(drop=True)

fallback = threshold_df.dropna(subset=["robust_score"]).copy()
if len(fallback) == 0:
    raise ValueError("No valid thresholds found in validation set.")

best_threshold = float(
    fallback.sort_values(
        ["robust_score", "mean_target_return_15m", "hit_rate", "rows"],
        ascending=[False, False, False, False],
    ).iloc[0]["threshold"]
)


# =========================================================
# TEST EVALUATION
# =========================================================
test_eval = test_df.copy()
test_eval["pred_return_15m"] = test_pred

test_result = evaluate_threshold(
    df=test_eval,
    pred_col="pred_return_15m",
    threshold=best_threshold,
    apply_filters=USE_EXECUTION_FILTERS,
    vol_filter_quantile=VOL_FILTER_QUANTILE,
)

test_mask = test_eval["pred_return_15m"] >= best_threshold
if USE_EXECUTION_FILTERS:
    test_vol_cut = test_eval["vol_30"].quantile(VOL_FILTER_QUANTILE)
    test_mask &= test_eval["vol_30"] >= test_vol_cut
    test_mask &= test_eval["trend_ok"] == 1
    test_mask &= test_eval["breakout_ok"] == 1

test_eval["take_trade"] = test_mask.astype(int)
test_taken = test_eval[test_eval["take_trade"] == 1].copy()

val_corr = pd.Series(y_val).corr(pd.Series(val_pred))
test_corr = pd.Series(y_test).corr(pd.Series(test_pred))

metrics = {
    "train_rows": int(len(train_df)),
    "val_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "val_mae": float(mean_absolute_error(y_val, val_pred)),
    "test_mae": float(mean_absolute_error(y_test, test_pred)),
    "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
    "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
    "val_corr": float(val_corr) if pd.notna(val_corr) else None,
    "test_corr": float(test_corr) if pd.notna(test_corr) else None,
    "best_threshold_from_val": float(best_threshold),
    "use_execution_filters": USE_EXECUTION_FILTERS,
    "vol_filter_quantile": VOL_FILTER_QUANTILE if USE_EXECUTION_FILTERS else None,
    "test_take_rate_at_best_threshold": float(test_result["take_rate"]),
    "test_rows_taken": int(test_result["rows"]),
    "test_mean_return_taken": float(test_result["mean_target_return_15m"]) if pd.notna(test_result["mean_target_return_15m"]) else np.nan,
    "test_hit_rate_taken": float(test_result["hit_rate"]) if pd.notna(test_result["hit_rate"]) else np.nan,
    "test_mean_pred_return_taken": float(test_taken["pred_return_15m"].mean()) if len(test_taken) else np.nan,
    "pred_distribution": {
        "val_mean": float(np.mean(val_pred)),
        "val_std": float(np.std(val_pred)),
        "val_min": float(np.min(val_pred)),
        "val_max": float(np.max(val_pred)),
        "test_mean": float(np.mean(test_pred)),
        "test_std": float(np.std(test_pred)),
        "test_min": float(np.min(test_pred)),
        "test_max": float(np.max(test_pred)),
    },
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
    scoring="neg_mean_squared_error",
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
full_pred = pipeline.predict(df[feature_cols])
df["pred_return_15m"] = full_pred

full_mask = df["pred_return_15m"] >= best_threshold
if USE_EXECUTION_FILTERS:
    full_vol_cut = df["vol_30"].quantile(VOL_FILTER_QUANTILE)
    full_mask &= df["vol_30"] >= full_vol_cut
    full_mask &= df["trend_ok"] == 1
    full_mask &= df["breakout_ok"] == 1

df["take_trade"] = full_mask.astype(int)
df["realized_pnl_if_taken"] = np.where(df["take_trade"] == 1, df["target_return_15m"], 0.0)

scored_cols = [
    "timestamp",
    "close",
    "target_return_15m",
    "pred_return_15m",
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
    "best_threshold": best_threshold,
    "timezone": TIMEZONE,
    "target_horizon_minutes": TARGET_HORIZON_MINUTES,
    "use_execution_filters": USE_EXECUTION_FILTERS,
    "vol_filter_quantile": VOL_FILTER_QUANTILE,
    "breakout_dist_high_15_min": BREAKOUT_DIST_HIGH_15_MIN,
    "trend_dist_sma_15_min": TREND_DIST_SMA_15_MIN,
    "trend_ret_5_min": TREND_RET_5_MIN,
    "trend_slope_sma_15_min": TREND_SLOPE_SMA_15_MIN,
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
    "val_mae",
    "test_mae",
    "val_rmse",
    "test_rmse",
    "val_corr",
    "test_corr",
    "best_threshold_from_val",
    "test_take_rate_at_best_threshold",
    "test_rows_taken",
    "test_mean_return_taken",
    "test_hit_rate_taken",
]:
    print(f"{key}: {metrics[key]}")

print("\n=== TOP FEATURES ===")
print(feature_importance_df.head(15).to_string(index=False))

print("\n=== VALIDATION THRESHOLDS ===")
print(threshold_df.to_string(index=False))