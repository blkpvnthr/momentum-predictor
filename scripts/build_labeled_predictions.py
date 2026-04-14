from __future__ import annotations

import pandas as pd
from pathlib import Path

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = PROJECT_ROOT / "outputs" / "signals" / "predictions.csv"
PRICE_PATH = PROJECT_ROOT / "data" / "market_data.csv"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "training" / "labeled_predictions.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# CONFIG
# =========================================================
HORIZON_MINUTES = 15


# =========================================================
# LOAD
# =========================================================
if not PRED_PATH.exists():
    raise FileNotFoundError(f"Missing predictions: {PRED_PATH}")

if not PRICE_PATH.exists():
    raise FileNotFoundError(f"Missing market data: {PRICE_PATH}")

preds = pd.read_csv(PRED_PATH)
prices = pd.read_csv(PRICE_PATH)

# =========================================================
# CLEAN TIMESTAMPS
# =========================================================
preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True, errors="coerce")
prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True, errors="coerce")

preds = preds.dropna(subset=["timestamp"]).copy()
prices = prices.dropna(subset=["timestamp"]).copy()

prices = prices.sort_values("timestamp").reset_index(drop=True)

print("preds rows:", len(preds))
print("prices rows:", len(prices))

print("preds timestamp sample:")
print(preds["timestamp"].head())

print("prices timestamp sample:")
print(prices["timestamp"].head())

print("preds min/max:", preds["timestamp"].min(), preds["timestamp"].max())
print("prices min/max:", prices["timestamp"].min(), prices["timestamp"].max())


# =========================================================
# BUILD FUTURE RETURN (TIMESTAMP-BASED)
# =========================================================
prices = prices.sort_values("timestamp").reset_index(drop=True)

# shift forward WITHOUT touching timestamps
prices["future_close"] = prices["close"].shift(-HORIZON_MINUTES)

# IMPORTANT: drop last rows BEFORE computing
prices = prices.dropna(subset=["future_close"]).copy()

prices["actual_return_15m"] = (
    prices["future_close"] / prices["close"] - 1.0
)

labels = prices[["timestamp", "actual_return_15m"]].copy()

# keep original timestamp column
price_frame = prices[["timestamp", "close"]].copy()

future_frame = price_frame.copy()
future_frame["timestamp"] = future_frame["timestamp"] - pd.Timedelta(minutes=HORIZON_MINUTES)
future_frame = future_frame.rename(columns={"close": "future_close"})

prices = price_frame.merge(
    future_frame,
    on="timestamp",
    how="left",
)

prices["actual_return_15m"] = (
    prices["future_close"] / prices["close"] - 1.0
)

labels = prices[["timestamp", "actual_return_15m"]].dropna().copy()

# =========================================================
# ALIGN (SAFE MERGE)
# =========================================================
preds["ts_key"] = preds["timestamp"].dt.floor("min")
labels["ts_key"] = labels["timestamp"].dt.floor("min")

df = preds.merge(
    labels[["ts_key", "actual_return_15m"]],
    on="ts_key",
    how="inner",
)

df["timestamp"] = df["ts_key"]
df = df.drop(columns=["ts_key"])

print(df[["timestamp", "pred_return_15m", "actual_return_15m"]].head(20))

print("CORRELATION CHECK:")
print(df["pred_return_15m"].corr(df["actual_return_15m"]))

# =========================================================
# SAVE
# =========================================================
print(f"Merged rows: {len(df)}")

if len(df) == 0:
    raise ValueError("No overlap between predictions and price data.")

df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved labeled predictions → {OUTPUT_PATH}")
print(df.head())