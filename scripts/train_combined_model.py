from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "combined_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "combined_model.joblib"
METRICS_PATH = OUTPUT_DIR / "combined_model_metrics.json"
SCORED_PATH = OUTPUT_DIR / "combined_model_scored.csv"


# =========================================================
# CONFIG
# =========================================================
TIMEZONE = "America/New_York"
TARGET_HORIZON_MINUTES = 15
COST_BUFFER = 0.0005

TEST_SIZE_FRACTION = 0.20
VALIDATION_SIZE_FRACTION = 0.15

MINUTES_FROM_OPEN = 0
MINUTES_TO_CLOSE_MIN = TARGET_HORIZON_MINUTES


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

    if "open" not in df.columns:
        if "o" in df.columns:
            rename_map["o"] = "open"
        elif "Open" in df.columns:
            rename_map["Open"] = "open"

    if "high" not in df.columns:
        if "h" in df.columns:
            rename_map["h"] = "high"
        elif "High" in df.columns:
            rename_map["High"] = "high"

    if "low" not in df.columns:
        if "l" in df.columns:
            rename_map["l"] = "low"
        elif "Low" in df.columns:
            rename_map["Low"] = "low"

    if "volume" not in df.columns:
        if "v" in df.columns:
            rename_map["v"] = "volume"
        elif "Volume" in df.columns:
            rename_map["Volume"] = "volume"

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


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if not {"high", "low", "close"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period, min_periods=period).mean()


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

    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_15"] = close.pct_change(15)
    df["ret_30"] = close.pct_change(30)

    df["mom_3"] = close / close.shift(3) - 1.0
    df["mom_5"] = close / close.shift(5) - 1.0
    df["mom_15"] = close / close.shift(15) - 1.0
    df["mom_30"] = close / close.shift(30) - 1.0

    df["vol_5"] = df["ret_1"].rolling(5, min_periods=5).std()
    df["vol_15"] = df["ret_1"].rolling(15, min_periods=15).std()
    df["vol_30"] = df["ret_1"].rolling(30, min_periods=30).std()

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

    df["rolling_high_15"] = close.rolling(15, min_periods=15).max()
    df["rolling_low_15"] = close.rolling(15, min_periods=15).min()
    df["pos_in_range_15"] = (close - df["rolling_low_15"]) / (
        (df["rolling_high_15"] - df["rolling_low_15"]) + 1e-12
    )

    df["rsi_14"] = compute_rsi(close, period=14)

    if {"open", "high", "low"}.issubset(df.columns):
        df["bar_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-12)
        df["body"] = (df["close"] - df["open"]) / (df["open"] + 1e-12)
        df["close_loc"] = (df["close"] - df["low"]) / ((df["high"] - df["low"]) + 1e-12)
        df["atr_14"] = compute_atr(df, period=14)
        df["atr_pct_14"] = df["atr_14"] / (df["close"] + 1e-12)

    if "volume" in df.columns:
        df["vol_chg_1"] = df["volume"].pct_change(1)
        df["vol_chg_5"] = df["volume"].pct_change(5)
        pv = df["close"] * df["volume"]
        df["cum_pv"] = pv.groupby(df["session_date"]).cumsum()
        df["cum_vol"] = df["volume"].groupby(df["session_date"]).cumsum()
        df["vwap"] = df["cum_pv"] / (df["cum_vol"] + 1e-12)
        df["dist_vwap"] = df["close"] / df["vwap"] - 1.0

    return df


def add_targets(df: pd.DataFrame, horizon: int, cost_buffer: float) -> pd.DataFrame:
    future_close = df["close"].shift(-horizon)
    df["target_return_15m"] = future_close / df["close"] - 1.0
    df["target_long_profit_15m"] = (df["target_return_15m"] > cost_buffer).astype(int)
    return df


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
df = add_targets(df, TARGET_HORIZON_MINUTES, COST_BUFFER)

df = df[df["minutes_from_open"] >= MINUTES_FROM_OPEN].copy()
df = df[df["minutes_to_close"] >= MINUTES_TO_CLOSE_MIN].copy()
df = df.dropna(subset=["target_return_15m"]).copy()

print("Columns in df:")
print(df.columns.tolist())
print("Row count before feature filtering:", len(df))

candidate_feature_cols = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_15",
    "ret_30",
    "mom_3",
    "mom_5",
    "mom_15",
    "mom_30",
    "vol_5",
    "vol_15",
    "vol_30",
    "dist_sma_5",
    "dist_sma_15",
    "dist_sma_30",
    "dist_sma_60",
    "slope_sma_5",
    "slope_sma_15",
    "slope_sma_30",
    "pos_in_range_15",
    "rsi_14",
    "bar_range",
    "body",
    "close_loc",
    "atr_pct_14",
    "vol_chg_1",
    "vol_chg_5",
    "dist_vwap",
    "hour",
    "minute",
    "day_of_week",
    "minutes_from_open",
    "minutes_to_close",
    "is_opening_window",
    "is_midday",
    "is_power_hour",
]

feature_cols = [
    col for col in candidate_feature_cols
    if col in df.columns and not df[col].isna().all()
]

target_cls_col = "target_long_profit_15m"
target_reg_col = "target_return_15m"

if not feature_cols:
    raise ValueError("No usable feature columns were created.")

df = df.dropna(subset=[target_cls_col, target_reg_col]).copy()
df = df.dropna(subset=feature_cols).copy()

print(f"Using {len(feature_cols)} features:")
print(feature_cols)

if len(df) < 1000:
    raise ValueError(f"Not enough rows after feature engineering: {len(df)}")


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

y_train_cls = train_df[target_cls_col]
y_val_cls = val_df[target_cls_col]
y_test_cls = test_df[target_cls_col]

y_train_reg = train_df[target_reg_col]
y_val_reg = val_df[target_reg_col]
y_test_reg = test_df[target_reg_col]


# =========================================================
# SAMPLE WEIGHTS
# =========================================================
train_weights_cls = np.clip(train_df["target_return_15m"].abs() / 0.001, 1.0, 10.0)
train_weights_reg = np.clip(train_df["target_return_15m"].abs() / 0.001, 1.0, 10.0)


# =========================================================
# MODELS
# =========================================================
classifier = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=400,
                max_depth=6,
                min_samples_leaf=50,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ]
)

regressor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=400,
                max_depth=6,
                min_samples_leaf=50,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ]
)

classifier.fit(X_train, y_train_cls, model__sample_weight=train_weights_cls)
regressor.fit(X_train, y_train_reg, model__sample_weight=train_weights_reg)


# =========================================================
# PREDICTIONS
# =========================================================
val_prob = classifier.predict_proba(X_val)[:, 1]
test_prob = classifier.predict_proba(X_test)[:, 1]

val_pred_return = regressor.predict(X_val)
test_pred_return = regressor.predict(X_test)

val_pred_cls = (val_prob >= 0.5).astype(int)
test_pred_cls = (test_prob >= 0.5).astype(int)

val_score = val_prob * val_pred_return
test_score = test_prob * test_pred_return


# =========================================================
# VALIDATION THRESHOLD SWEEP
# =========================================================
threshold_rows: list[dict] = []

candidate_thresholds = np.quantile(val_score, [0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
candidate_thresholds = np.unique(candidate_thresholds)

val_eval = val_df.copy()
val_eval["prob_long_profit_15m"] = val_prob
val_eval["pred_return_15m"] = val_pred_return
val_eval["combined_score"] = val_score

for threshold in candidate_thresholds:
    sub = val_eval[val_eval["combined_score"] >= threshold].copy()

    mean_ret = float(sub["target_return_15m"].mean()) if len(sub) else np.nan
    hit_rate = float((sub["target_return_15m"] > COST_BUFFER).mean()) if len(sub) else np.nan
    rows = int(len(sub))
    take_rate = float(rows / len(val_eval)) if len(val_eval) else np.nan
    robust_score = float(mean_ret * np.sqrt(rows)) if rows > 0 and pd.notna(mean_ret) else np.nan

    threshold_rows.append(
        {
            "threshold": float(threshold),
            "rows": rows,
            "take_rate": take_rate,
            "mean_target_return_15m": mean_ret,
            "hit_rate": hit_rate,
            "robust_score": robust_score,
        }
    )

threshold_df = pd.DataFrame(threshold_rows).sort_values("threshold").reset_index(drop=True)

valid_threshold_df = threshold_df.dropna(subset=["robust_score"]).copy()
if len(valid_threshold_df) == 0:
    best_threshold = float(np.quantile(val_score, 0.80))
else:
    best_threshold = float(
        valid_threshold_df.sort_values(
            ["robust_score", "mean_target_return_15m", "hit_rate", "rows"],
            ascending=[False, False, False, False],
        ).iloc[0]["threshold"]
    )


# =========================================================
# TEST EVALUATION
# =========================================================
test_eval = test_df.copy()
test_eval["prob_long_profit_15m"] = test_prob
test_eval["pred_return_15m"] = test_pred_return
test_eval["combined_score"] = test_score
test_eval["take_trade"] = (test_eval["combined_score"] >= best_threshold).astype(int)

test_taken = test_eval[test_eval["take_trade"] == 1].copy()

cls_metrics = {
    "val_accuracy_at_0_5": float(accuracy_score(y_val_cls, val_pred_cls)),
    "test_accuracy_at_0_5": float(accuracy_score(y_test_cls, test_pred_cls)),
    "val_precision_at_0_5": float(precision_score(y_val_cls, val_pred_cls, zero_division=0)),
    "test_precision_at_0_5": float(precision_score(y_test_cls, test_pred_cls, zero_division=0)),
    "val_recall_at_0_5": float(recall_score(y_val_cls, val_pred_cls, zero_division=0)),
    "test_recall_at_0_5": float(recall_score(y_test_cls, test_pred_cls, zero_division=0)),
    "val_roc_auc": float(roc_auc_score(y_val_cls, val_prob)) if len(np.unique(y_val_cls)) > 1 else None,
    "test_roc_auc": float(roc_auc_score(y_test_cls, test_prob)) if len(np.unique(y_test_cls)) > 1 else None,
    "test_classification_report_at_0_5": classification_report(y_test_cls, test_pred_cls, zero_division=0),
}

reg_metrics = {
    "val_mae_return": float(mean_absolute_error(y_val_reg, val_pred_return)),
    "test_mae_return": float(mean_absolute_error(y_test_reg, test_pred_return)),
    "val_rmse_return": float(np.sqrt(mean_squared_error(y_val_reg, val_pred_return))),
    "test_rmse_return": float(np.sqrt(mean_squared_error(y_test_reg, test_pred_return))),
    "val_pred_actual_corr": float(pd.Series(val_pred_return).corr(pd.Series(y_val_reg))),
    "test_pred_actual_corr": float(pd.Series(test_pred_return).corr(pd.Series(y_test_reg))),
}

combined_metrics = {
    "best_threshold_from_val": float(best_threshold),
    "test_take_rate_at_best_threshold": float(test_eval["take_trade"].mean()),
    "test_rows_taken": int(len(test_taken)),
    "test_mean_return_taken": float(test_taken["target_return_15m"].mean()) if len(test_taken) else np.nan,
    "test_hit_rate_taken": float((test_taken["target_return_15m"] > COST_BUFFER).mean()) if len(test_taken) else np.nan,
    "test_mean_prob_taken": float(test_taken["prob_long_profit_15m"].mean()) if len(test_taken) else np.nan,
    "test_mean_pred_return_taken": float(test_taken["pred_return_15m"].mean()) if len(test_taken) else np.nan,
    "test_mean_combined_score_taken": float(test_taken["combined_score"].mean()) if len(test_taken) else np.nan,
}

metrics = {
    "train_rows": int(len(train_df)),
    "val_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "positive_rate_train": float(y_train_cls.mean()),
    "positive_rate_val": float(y_val_cls.mean()),
    "positive_rate_test": float(y_test_cls.mean()),
    **cls_metrics,
    **reg_metrics,
    **combined_metrics,
}


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
perm_cls = permutation_importance(
    classifier,
    X_val,
    y_val_cls,
    n_repeats=5,
    random_state=42,
    scoring="roc_auc",
)

feature_importance_cls_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance_mean": perm_cls.importances_mean,
        "importance_std": perm_cls.importances_std,
    }
).sort_values("importance_mean", ascending=False)

perm_reg = permutation_importance(
    regressor,
    X_val,
    y_val_reg,
    n_repeats=5,
    random_state=42,
    scoring="neg_mean_squared_error",
)

feature_importance_reg_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance_mean": perm_reg.importances_mean,
        "importance_std": perm_reg.importances_std,
    }
).sort_values("importance_mean", ascending=False)

metrics["top_features_classifier"] = feature_importance_cls_df.head(15).to_dict(orient="records")
metrics["top_features_regressor"] = feature_importance_reg_df.head(15).to_dict(orient="records")
metrics["validation_thresholds"] = threshold_rows


# =========================================================
# SCORE FULL DATA
# =========================================================
full_prob = classifier.predict_proba(df[feature_cols])[:, 1]
full_pred_return = regressor.predict(df[feature_cols])
full_score = full_prob * full_pred_return

df["prob_long_profit_15m"] = full_prob
df["pred_return_15m"] = full_pred_return
df["combined_score"] = full_score
df["take_trade"] = (df["combined_score"] >= best_threshold).astype(int)
df["realized_pnl_if_taken"] = np.where(df["take_trade"] == 1, df["target_return_15m"], 0.0)

scored_cols = [
    "timestamp",
    "close",
    "target_return_15m",
    "target_long_profit_15m",
    "prob_long_profit_15m",
    "pred_return_15m",
    "combined_score",
    "take_trade",
    "realized_pnl_if_taken",
    *feature_cols,
]
scored_df = df[scored_cols].copy()


# =========================================================
# SAVE
# =========================================================
bundle = {
    "classifier": classifier,
    "regressor": regressor,
    "feature_cols": feature_cols,
    "target_cls_col": target_cls_col,
    "target_reg_col": target_reg_col,
    "cost_buffer": COST_BUFFER,
    "best_threshold": best_threshold,
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
    "val_pred_actual_corr",
    "test_pred_actual_corr",
    "best_threshold_from_val",
    "test_take_rate_at_best_threshold",
    "test_rows_taken",
    "test_mean_return_taken",
    "test_hit_rate_taken",
]:
    print(f"{key}: {metrics[key]}")

print("\n=== TOP CLASSIFIER FEATURES ===")
print(feature_importance_cls_df.head(15).to_string(index=False))

print("\n=== TOP REGRESSOR FEATURES ===")
print(feature_importance_reg_df.head(15).to_string(index=False))

print("\n=== VALIDATION THRESHOLDS ===")
print(threshold_df.to_string(index=False))