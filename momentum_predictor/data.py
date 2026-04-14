# scripts/download_prices.py

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# =========================================================
# PATH SETUP
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

OUTPUT_PATH = PROJECT_ROOT / "data" / "market_data.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError("Missing Alpaca credentials in .env")

# =========================================================
# CLIENT
# =========================================================
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# =========================================================
# CONFIG
# =========================================================
SYMBOL = "QQQ"

START = datetime(2026, 3, 1)
END = datetime(2026, 4, 1)

# =========================================================
# REQUEST
# =========================================================
request = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Minute,
    start=START,
    end=END,
)

# =========================================================
# FETCH DATA
# =========================================================
bars = client.get_stock_bars(request).df

# Handle multi-index (symbol, timestamp)
if isinstance(bars.index, pd.MultiIndex):
    bars = bars.reset_index()

# Filter symbol if needed
if "symbol" in bars.columns:
    bars = bars[bars["symbol"] == SYMBOL].copy()

# =========================================================
# CLEAN DATA
# =========================================================
bars = bars.rename(columns={
    "timestamp": "timestamp",
    "close": "close",
})

df = bars[["timestamp", "close"]].sort_values("timestamp")

# =========================================================
# SAVE
# =========================================================
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Saved {len(df)} rows to {OUTPUT_PATH}")