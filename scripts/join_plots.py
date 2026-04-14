from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

PRED_PATH = PROJECT_ROOT / "outputs" / "signals" / "predictions.csv"
PRICE_PATH = PROJECT_ROOT / "data" / "market_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELED_PATH = OUTPUT_DIR / "labeled_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "report_summary.csv"
SIGNAL_REPORT_PATH = OUTPUT_DIR / "signal_report.csv"
CONF_BUCKET_PATH = OUTPUT_DIR / "confidence_buckets.csv"


# =========================================================
# CONFIG
# =========================================================
HORIZONS = (5, 15, 30)
CONF_BUCKETS = 10
TIMEZONE = "America/New_York"
MARKET_CLOSE_CUTOFF = pd.Timestamp("15:30").time()

# Keep both long and short behavior visible unless you explicitly want otherwise.
DISABLE_SHORTS = False


# =========================================================
# HELPERS
# =========================================================
def require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def normalize_price_columns(prices: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}

    if "close" not in prices.columns:
        if "c" in prices.columns:
            rename_map["c"] = "close"
        elif "Close" in prices.columns:
            rename_map["Close"] = "close"
        elif "close_price" in prices.columns:
            rename_map["close_price"] = "close"

    if "timestamp" not in prices.columns:
        if "time" in prices.columns:
            rename_map["time"] = "timestamp"
        elif "datetime" in prices.columns:
            rename_map["datetime"] = "timestamp"
        elif "Datetime" in prices.columns:
            rename_map["Datetime"] = "timestamp"

    if rename_map:
        prices = prices.rename(columns=rename_map)

    require_columns(prices, ["timestamp", "close"], "prices")
    return prices


def parse_timestamp_series(series: pd.Series, timezone: str) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_convert(timezone)


def safe_qcut(series: pd.Series, q: int) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="object")

    try:
        return pd.qcut(series, q=q, duplicates="drop")
    except ValueError:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="object")


def mean_if_exists(df: pd.DataFrame, col: str) -> float:
    if len(df) == 0 or col not in df.columns:
        return np.nan
    return float(df[col].mean())


# =========================================================
# LOAD
# =========================================================
if not PRED_PATH.exists():
    raise FileNotFoundError(f"Missing predictions file: {PRED_PATH}")

if not PRICE_PATH.exists():
    raise FileNotFoundError(f"Missing market data file: {PRICE_PATH}")

preds = pd.read_csv(PRED_PATH)
prices = pd.read_csv(PRICE_PATH)

require_columns(
    preds,
    [
        "timestamp",
        "signal",
        "confidence",
        "pred_return_5m",
        "pred_return_15m",
        "pred_return_30m",
    ],
    "predictions",
)

prices = normalize_price_columns(prices)

preds["timestamp"] = parse_timestamp_series(preds["timestamp"], TIMEZONE)
prices["timestamp"] = parse_timestamp_series(prices["timestamp"], TIMEZONE)

preds = preds.dropna(subset=["timestamp"]).copy()
prices = prices.dropna(subset=["timestamp", "close"]).copy()

preds = preds.sort_values("timestamp").reset_index(drop=True)
prices = prices.sort_values("timestamp").reset_index(drop=True)

prices = prices[["timestamp", "close"]].drop_duplicates(subset=["timestamp"], keep="last").copy()

print("preds rows:", len(preds))
print("prices rows:", len(prices))
print("preds min/max:", preds["timestamp"].min(), preds["timestamp"].max())
print("prices min/max:", prices["timestamp"].min(), prices["timestamp"].max())

if DISABLE_SHORTS:
    preds = preds[preds["signal"] != "SHORT"].copy()

# Avoid horizons that obviously run into the close
preds = preds[preds["timestamp"].dt.time < MARKET_CLOSE_CUTOFF].copy()

if preds.empty:
    raise ValueError("No prediction rows remain after filtering.")


# =========================================================
# ENTRY JOIN
# Use last known price at or before prediction timestamp
# =========================================================
entry_prices = prices.rename(
    columns={
        "timestamp": "entry_timestamp",
        "close": "entry_close",
    }
).sort_values("entry_timestamp")

labeled = pd.merge_asof(
    preds.sort_values("timestamp"),
    entry_prices,
    left_on="timestamp",
    right_on="entry_timestamp",
    direction="backward",
)

labeled = labeled.dropna(subset=["entry_close"]).copy()

if labeled.empty:
    raise ValueError("No rows remained after entry price join.")


# =========================================================
# FUTURE PRICE JOINS
# Use first available price at or after each horizon target
# =========================================================
for minutes in HORIZONS:
    target_col = f"target_timestamp_{minutes}m"
    future_ts_col = f"future_timestamp_{minutes}m"
    future_px_col = f"future_close_{minutes}m"
    actual_ret_col = f"actual_return_{minutes}m"

    labeled[target_col] = labeled["timestamp"] + pd.Timedelta(minutes=minutes)

    future_prices = prices.rename(
        columns={
            "timestamp": future_ts_col,
            "close": future_px_col,
        }
    ).sort_values(future_ts_col)

    future_join = pd.merge_asof(
        labeled[[target_col]].sort_values(target_col),
        future_prices,
        left_on=target_col,
        right_on=future_ts_col,
        direction="forward",
    )

    future_join = future_join[[target_col, future_ts_col, future_px_col]]
    labeled = labeled.merge(future_join, on=target_col, how="left")

    labeled = labeled[
        labeled[future_ts_col].notna() &
        (labeled[future_ts_col] > labeled["timestamp"])
    ].copy()

    labeled[actual_ret_col] = (
        labeled[future_px_col] / labeled["entry_close"] - 1.0
    )

if labeled.empty:
    raise ValueError("No rows remained after future price joins.")


# =========================================================
# DERIVED FIELDS
# =========================================================
# Keep original confidence and add a magnitude proxy separately
labeled["confidence_abs_pred_15m"] = labeled["pred_return_15m"].abs()

# Vectorized trade PnL
labeled["pnl_5m"] = 0.0
labeled["pnl_15m"] = 0.0
labeled["pnl_30m"] = 0.0

long_mask = labeled["signal"] == "LONG"
short_mask = labeled["signal"] == "SHORT"

labeled.loc[long_mask, "pnl_5m"] = labeled.loc[long_mask, "actual_return_5m"]
labeled.loc[long_mask, "pnl_15m"] = labeled.loc[long_mask, "actual_return_15m"]
labeled.loc[long_mask, "pnl_30m"] = labeled.loc[long_mask, "actual_return_30m"]

labeled.loc[short_mask, "pnl_5m"] = -labeled.loc[short_mask, "actual_return_5m"]
labeled.loc[short_mask, "pnl_15m"] = -labeled.loc[short_mask, "actual_return_15m"]
labeled.loc[short_mask, "pnl_30m"] = -labeled.loc[short_mask, "actual_return_30m"]

# Vectorized direction correctness
labeled["correct_direction"] = np.nan
labeled.loc[long_mask, "correct_direction"] = (
    labeled.loc[long_mask, "actual_return_15m"] > 0
).astype(float)
labeled.loc[short_mask, "correct_direction"] = (
    labeled.loc[short_mask, "actual_return_15m"] < 0
).astype(float)

# Prediction errors
labeled["prediction_error_5m"] = labeled["actual_return_5m"] - labeled["pred_return_5m"]
labeled["prediction_error_15m"] = labeled["actual_return_15m"] - labeled["pred_return_15m"]
labeled["prediction_error_30m"] = labeled["actual_return_30m"] - labeled["pred_return_30m"]
labeled["abs_prediction_error_15m"] = labeled["prediction_error_15m"].abs()

trade_df = labeled[labeled["signal"].isin(["LONG", "SHORT"])].copy()


# =========================================================
# SUMMARY REPORT
# =========================================================
pred_actual_corr_15m_all = (
    labeled["pred_return_15m"].corr(labeled["actual_return_15m"])
    if len(labeled) > 1 else np.nan
)

pred_actual_corr_15m_trades = (
    trade_df["pred_return_15m"].corr(trade_df["actual_return_15m"])
    if len(trade_df) > 1 else np.nan
)

summary = {
    "rows_total": int(len(labeled)),
    "rows_with_entry_price": int(labeled["entry_close"].notna().sum()),
    "trade_rows": int(len(trade_df)),
    "signal_long_count": int((labeled["signal"] == "LONG").sum()),
    "signal_short_count": int((labeled["signal"] == "SHORT").sum()),
    "signal_no_trade_count": int((labeled["signal"] == "NO_TRADE").sum()),
    "mean_pred_return_5m": mean_if_exists(labeled, "pred_return_5m"),
    "mean_pred_return_15m": mean_if_exists(labeled, "pred_return_15m"),
    "mean_pred_return_30m": mean_if_exists(labeled, "pred_return_30m"),
    "mean_actual_return_5m": mean_if_exists(labeled, "actual_return_5m"),
    "mean_actual_return_15m": mean_if_exists(labeled, "actual_return_15m"),
    "mean_actual_return_30m": mean_if_exists(labeled, "actual_return_30m"),
    "trade_hit_rate_15m": mean_if_exists(trade_df, "correct_direction"),
    "trade_mean_pnl_15m": mean_if_exists(trade_df, "pnl_15m"),
    "trade_sum_pnl_15m": float(trade_df["pnl_15m"].sum()) if len(trade_df) else np.nan,
    "pred_actual_corr_15m_all": float(pred_actual_corr_15m_all) if pd.notna(pred_actual_corr_15m_all) else np.nan,
    "pred_actual_corr_15m_trades": float(pred_actual_corr_15m_trades) if pd.notna(pred_actual_corr_15m_trades) else np.nan,
    "mean_abs_prediction_error_15m": mean_if_exists(trade_df, "abs_prediction_error_15m"),
    "pct_zero_actual_return_15m_all": float((labeled["actual_return_15m"] == 0).mean()) if len(labeled) else np.nan,
    "pct_zero_pnl_15m_trades": float((trade_df["pnl_15m"] == 0).mean()) if len(trade_df) else np.nan,
}

summary_df = pd.DataFrame([summary])


# =========================================================
# SIGNAL REPORT
# =========================================================
signal_rows: list[dict] = []

for signal_name, grp in labeled.groupby("signal", dropna=False):
    is_trade_signal = signal_name in {"LONG", "SHORT"}

    signal_rows.append(
        {
            "signal": signal_name,
            "count": int(len(grp)),
            "mean_confidence": mean_if_exists(grp, "confidence"),
            "mean_confidence_abs_pred_15m": mean_if_exists(grp, "confidence_abs_pred_15m"),
            "mean_pred_return_15m": mean_if_exists(grp, "pred_return_15m"),
            "mean_actual_return_15m": mean_if_exists(grp, "actual_return_15m"),
            "mean_pnl_15m": mean_if_exists(grp, "pnl_15m"),
            "hit_rate_15m": mean_if_exists(grp if is_trade_signal else pd.DataFrame(), "correct_direction"),
        }
    )

signal_report_df = pd.DataFrame(signal_rows).sort_values("signal").reset_index(drop=True)


# =========================================================
# CONFIDENCE BUCKET REPORT
# =========================================================
if len(trade_df) > 0:
    trade_df["confidence_bucket"] = safe_qcut(trade_df["confidence_abs_pred_15m"], CONF_BUCKETS)

    conf_report_df = (
        trade_df.groupby("confidence_bucket", dropna=False, observed=False)
        .apply(
            lambda g: pd.Series(
                {
                    "count": int(len(g)),
                    "mean_confidence_original": g["confidence"].mean(),
                    "mean_confidence_abs_pred_15m": g["confidence_abs_pred_15m"].mean(),
                    "mean_pred_return_15m": g["pred_return_15m"].mean(),
                    "mean_actual_return_15m": g["actual_return_15m"].mean(),
                    "mean_pnl_15m": g["pnl_15m"].mean(),
                    "hit_rate_15m": g["correct_direction"].mean(),
                    "mean_abs_prediction_error_15m": g["abs_prediction_error_15m"].mean(),
                }
            )
        )
        .reset_index()
    )
else:
    conf_report_df = pd.DataFrame(
        columns=[
            "confidence_bucket",
            "count",
            "mean_confidence_original",
            "mean_confidence_abs_pred_15m",
            "mean_pred_return_15m",
            "mean_actual_return_15m",
            "mean_pnl_15m",
            "hit_rate_15m",
            "mean_abs_prediction_error_15m",
        ]
    )


# =========================================================
# SAVE
# =========================================================
labeled.to_csv(LABELED_PATH, index=False)
summary_df.to_csv(SUMMARY_PATH, index=False)
signal_report_df.to_csv(SIGNAL_REPORT_PATH, index=False)
conf_report_df.to_csv(CONF_BUCKET_PATH, index=False)

print(f"Saved labeled data to: {LABELED_PATH}")
print(f"Saved summary report to: {SUMMARY_PATH}")
print(f"Saved signal report to: {SIGNAL_REPORT_PATH}")
print(f"Saved confidence bucket report to: {CONF_BUCKET_PATH}")

print("\n=== SUMMARY ===")
print(summary_df.to_string(index=False))