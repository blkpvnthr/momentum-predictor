#!/usr/bin/env python3

"""
correlationNASDAQ.py

Find NASDAQ-listed stocks priced at or below $75 that are good candidates for a
QQQ/TQQQ-style trading universe by combining correlation, beta, volatility,
momentum persistence, liquidity, signal-to-noise, and trend quality.

Final ranking is prioritized by liquidity:
    1. avg_dollar_volume
    2. score
    3. corr

Run with:
    python correlationNASDAQ.py

Outputs:
    live/tqqq_nasdaq_prices.csv
    live/tqqq_nasdaq_volumes.csv
    live/tqqq_nasdaq_comovers.csv
    live/top30_symbols.txt
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


THIS_FILE = Path(__file__).resolve()
LIVE_DIR = THIS_FILE.parent
PROJECT_ROOT = LIVE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


# =========================
# CONFIG
# =========================
LOOKBACK_DAYS = 365 * 2
ROLLING_WINDOW = 60
CHUNK_SIZE = 75
PAUSE_SEC = 0.4
MIN_OBS = 120
VOLUME_LOOKBACK = 60

TARGET = "TQQQ"
BENCHMARK = "QQQ"
MAX_STOCK_PRICE = 75.0

OUT_CSV = LIVE_DIR / "tqqq_nasdaq_comovers.csv"
PRICES_CSV = LIVE_DIR / "tqqq_nasdaq_prices.csv"
VOLUMES_CSV = LIVE_DIR / "tqqq_nasdaq_volumes.csv"
TOP30_TXT = LIVE_DIR / "top30_symbols.txt"

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"

FACTOR_WEIGHTS = {
    "correlation_score": 0.18,
    "same_direction_score": 0.12,
    "up_capture_score": 0.08,
    "beta_score": 0.12,
    "volatility_score": 0.14,
    "momentum_persistence_score": 0.12,
    "liquidity_score": 0.12,
    "signal_to_noise_score": 0.06,
    "trend_quality_score": 0.06,
}


# =========================
# LOAD ENV
# =========================
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        f"Missing Alpaca credentials. Expected APCA_API_KEY_ID and "
        f"APCA_API_SECRET_KEY in {ENV_PATH}"
    )

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


# =========================
# GET NASDAQ SYMBOLS
# =========================
def get_nasdaq_constituents() -> list[str]:
    print("[setup] fetching NASDAQ-listed symbols...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(NASDAQ_LISTED_URL, headers=headers, timeout=20)
    resp.raise_for_status()

    lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError("Could not parse NASDAQ-listed symbol file.")

    data_lines = [line for line in lines if not line.startswith("File Creation Time")]
    df = pd.read_csv(StringIO("\n".join(data_lines)), sep="|")

    if "Symbol" not in df.columns:
        raise RuntimeError("NASDAQ symbol file does not contain a Symbol column.")

    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].astype(str).str.upper() == "N"].copy()

    cleaned: list[str] = []
    for raw_sym in df["Symbol"].tolist():
        if pd.isna(raw_sym):
            continue

        sym = str(raw_sym).strip().upper()

        if not sym or sym == "NAN":
            continue
        if "." in sym:
            continue

        cleaned.append(sym)

    for required in (TARGET, BENCHMARK):
        if required not in cleaned:
            cleaned.append(required)

    cleaned = sorted(set(cleaned))
    print(f"[setup] total NASDAQ-listed symbols: {len(cleaned)}")
    return cleaned


# =========================
# HELPERS
# =========================
def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def safe_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def normalize_series(
    s: pd.Series,
    clip_low: float | None = None,
    clip_high: float | None = None,
) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").copy()

    if clip_low is not None or clip_high is not None:
        s = s.clip(lower=clip_low, upper=clip_high)

    finite = s.replace([np.inf, -np.inf], np.nan)
    s_min = finite.min()
    s_max = finite.max()

    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(0.5, index=s.index, dtype=float)

    return (finite - s_min) / (s_max - s_min)


def ideal_beta_score(beta: pd.Series, target_low: float = 1.1, target_high: float = 3.0) -> pd.Series:
    out = []
    for val in beta:
        if pd.isna(val):
            out.append(np.nan)
        elif target_low <= val <= target_high:
            out.append(1.0)
        elif val < target_low:
            out.append(max(0.0, float(val) / target_low))
        else:
            out.append(max(0.0, target_high / float(val)))
    return pd.Series(out, index=beta.index, dtype=float)


def fetch_chunk(symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()

    if df.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", "close", "volume"])

    needed = {"symbol", "timestamp", "close", "volume"}
    missing = needed.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns from Alpaca bars response: {missing}")

    out = df[["symbol", "timestamp", "close", "volume"]].copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out


def compute_beta(symbol_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    x, y = symbol_returns.align(benchmark_returns, join="inner")
    pair = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(pair) < 2:
        return np.nan

    x_vals = pair["x"].to_numpy()
    y_vals = pair["y"].to_numpy()

    var_y = np.var(y_vals, ddof=1)
    if not np.isfinite(var_y) or var_y <= 0:
        return np.nan

    cov = np.cov(x_vals, y_vals, ddof=1)[0, 1]
    return float(cov / var_y)


def compute_annualized_volatility(returns: pd.Series) -> float:
    returns = safe_series(returns)
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * math.sqrt(252))


def compute_avg_dollar_volume(close: pd.Series, volume: pd.Series, lookback: int = VOLUME_LOOKBACK) -> float:
    c, v = close.align(volume, join="inner")
    pair = pd.DataFrame({"close": c, "volume": v}).dropna()

    if pair.empty:
        return np.nan

    tail = pair.tail(lookback)
    return float((tail["close"] * tail["volume"]).mean())


def compute_momentum_persistence(close: pd.Series) -> float:
    returns = safe_series(close.pct_change(fill_method=None))
    if len(returns) < 3:
        return np.nan

    signs = np.sign(returns.to_numpy())
    if len(signs) < 2:
        return np.nan

    return float((signs[1:] == signs[:-1]).mean())


def compute_signal_to_noise(close: pd.Series, window: int = 20) -> float:
    close = safe_series(close)
    if len(close) < window + 1:
        return np.nan

    vals: list[float] = []
    for i in range(window, len(close)):
        segment = close.iloc[i - window:i + 1]
        if len(segment) < window + 1:
            continue

        net_move = abs(segment.iloc[-1] / segment.iloc[0] - 1.0)
        path_noise = segment.pct_change(fill_method=None).std(ddof=1)

        if pd.notna(path_noise) and np.isfinite(path_noise) and path_noise > 0:
            vals.append(float(net_move / path_noise))

    if not vals:
        return np.nan
    return float(np.mean(vals))


def compute_trend_quality(close: pd.Series, short_window: int = 20, long_window: int = 50) -> float:
    close = safe_series(close)
    if len(close) < long_window + 5:
        return np.nan

    sma_short = close.rolling(short_window).mean()
    sma_long = close.rolling(long_window).mean()
    spread = ((sma_short - sma_long).abs() / close).dropna()

    if spread.empty:
        return np.nan
    return float(spread.mean())


# =========================
# DOWNLOAD DATA
# =========================
def download_prices(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)

    print(f"[download] {start.date()} → {end.date()}")

    all_frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for idx, chunk in enumerate(chunked(symbols, CHUNK_SIZE), start=1):
        print(f"[download] chunk {idx} ({len(chunk)} symbols)")

        try:
            df = fetch_chunk(chunk, start, end)

            if df.empty:
                print("[warning] empty chunk response, retrying symbol-by-symbol")
                raise RuntimeError("empty chunk response")

            all_frames.append(df)

        except Exception as e:
            print(f"[warning] chunk failed, retrying symbol-by-symbol: {e}")

            for sym in chunk:
                try:
                    df = fetch_chunk([sym], start, end)
                    if df.empty:
                        failed.append(sym)
                    else:
                        all_frames.append(df)
                except Exception as inner_e:
                    print(f"[warning] failed symbol {sym}: {inner_e}")
                    failed.append(sym)

                time.sleep(PAUSE_SEC)

        time.sleep(PAUSE_SEC)

    if not all_frames:
        raise RuntimeError("No price data downloaded from Alpaca.")

    data = pd.concat(all_frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")

    data = data.drop_duplicates(subset=["timestamp", "symbol"]).sort_values(["timestamp", "symbol"])

    prices = data.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    volumes = data.pivot(index="timestamp", columns="symbol", values="volume").sort_index()

    prices = prices.dropna(axis=1, how="all")
    volumes = volumes.dropna(axis=1, how="all")

    common_cols = sorted(set(prices.columns).intersection(volumes.columns))
    prices = prices[common_cols]
    volumes = volumes[common_cols]

    print(f"[download] completed. prices_shape={prices.shape} volumes_shape={volumes.shape}")

    for required in (TARGET, BENCHMARK):
        if required not in prices.columns:
            raise RuntimeError(f"{required} was not downloaded successfully.")

    return prices, volumes, sorted(set(failed))


# =========================
# METRICS
# =========================
def compute_metrics(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    print("[compute] calculating returns...")

    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    if TARGET not in returns.columns:
        raise RuntimeError(f"{TARGET} missing from returns matrix.")
    if BENCHMARK not in returns.columns:
        raise RuntimeError(f"{BENCHMARK} missing from returns matrix.")

    target_returns = returns[TARGET]
    benchmark_returns = returns[BENCHMARK]

    results: list[dict[str, float | int | str]] = []

    for symbol in returns.columns:
        if symbol in {TARGET, BENCHMARK}:
            continue

        stock_price_series = safe_series(prices[symbol])
        if stock_price_series.empty:
            continue

        latest_price = float(stock_price_series.iloc[-1])
        if latest_price > MAX_STOCK_PRICE:
            continue

        vol_series = safe_series(volumes[symbol])
        if vol_series.empty:
            continue

        avg_volume = float(vol_series.tail(VOLUME_LOOKBACK).mean())
        if not np.isfinite(avg_volume) or avg_volume <= 0:
            continue

        x_target, y_target = returns[symbol].align(target_returns, join="inner")
        pair_target = pd.DataFrame({"x": x_target, "y": y_target}).dropna()

        x_bench, y_bench = returns[symbol].align(benchmark_returns, join="inner")
        pair_bench = pd.DataFrame({"x": x_bench, "y": y_bench}).dropna()

        if len(pair_target) < MIN_OBS or len(pair_bench) < MIN_OBS:
            continue

        x_target_series = pair_target["x"]
        y_target_series = pair_target["y"]

        corr_to_target = x_target_series.corr(y_target_series)

        rolling_corr = x_target_series.rolling(ROLLING_WINDOW).corr(y_target_series)
        rolling_mean = rolling_corr.mean()
        rolling_std = rolling_corr.std()

        dir_pair = pair_target[(pair_target["x"] != 0) & (pair_target["y"] != 0)].copy()
        if len(dir_pair) < MIN_OBS:
            continue

        x_dir = np.sign(dir_pair["x"])
        y_dir = np.sign(dir_pair["y"])

        same_direction_pct = float((x_dir == y_dir).mean())

        target_up = dir_pair["y"] > 0
        target_down = dir_pair["y"] < 0

        up_up_pct = float(
            ((dir_pair["x"] > 0) & target_up).sum() / max(int(target_up.sum()), 1)
        )
        down_down_pct = float(
            ((dir_pair["x"] < 0) & target_down).sum() / max(int(target_down.sum()), 1)
        )

        symbol_returns = safe_series(returns[symbol])
        beta_to_qqq = compute_beta(symbol_returns, benchmark_returns)
        ann_vol = compute_annualized_volatility(symbol_returns)
        avg_dollar_volume = compute_avg_dollar_volume(prices[symbol], volumes[symbol])
        momentum_persistence = compute_momentum_persistence(prices[symbol])
        signal_to_noise = compute_signal_to_noise(prices[symbol])
        trend_quality = compute_trend_quality(prices[symbol])

        results.append(
            {
                "symbol": symbol,
                "price": latest_price,
                "avg_volume": avg_volume,
                "avg_dollar_volume": avg_dollar_volume,
                "n_obs": int(len(pair_target)),
                "direction_obs": int(len(dir_pair)),
                "same_direction_pct": same_direction_pct,
                "up_up_pct": up_up_pct,
                "down_down_pct": down_down_pct,
                "corr": float(corr_to_target) if pd.notna(corr_to_target) else np.nan,
                "rolling_mean": float(rolling_mean) if pd.notna(rolling_mean) else np.nan,
                "rolling_std": float(rolling_std) if pd.notna(rolling_std) else np.nan,
                "beta": float(beta_to_qqq) if pd.notna(beta_to_qqq) else np.nan,
                "annualized_volatility": ann_vol,
                "momentum_persistence": momentum_persistence,
                "signal_to_noise": signal_to_noise,
                "trend_quality": trend_quality,
            }
        )

    if not results:
        raise RuntimeError(
            "No valid symbol metrics were computed. "
            "This usually means the downloaded price/volume matrix is too sparse "
            "or the price filter is too restrictive."
        )

    ranked = pd.DataFrame(results)

    numeric_cols = [
        "price",
        "avg_volume",
        "avg_dollar_volume",
        "same_direction_pct",
        "up_up_pct",
        "down_down_pct",
        "corr",
        "rolling_mean",
        "rolling_std",
        "beta",
        "annualized_volatility",
        "momentum_persistence",
        "signal_to_noise",
        "trend_quality",
    ]
    for col in numeric_cols:
        ranked[col] = pd.to_numeric(ranked[col], errors="coerce")

    ranked = ranked.dropna(subset=["corr"])
    if ranked.empty:
        raise RuntimeError("Metric dataframe is empty after metric calculation.")

    ranked["correlation_score"] = normalize_series(ranked["corr"], clip_low=-1.0, clip_high=1.0)
    ranked["same_direction_score"] = normalize_series(
        ranked["same_direction_pct"], clip_low=0.0, clip_high=1.0
    )
    ranked["up_capture_score"] = normalize_series(
        ranked["up_up_pct"], clip_low=0.0, clip_high=1.0
    )

    raw_beta_score = ideal_beta_score(ranked["beta"], target_low=1.1, target_high=3.0)
    ranked["beta_score"] = normalize_series(raw_beta_score, clip_low=0.0, clip_high=1.0)

    vol_low = ranked["annualized_volatility"].quantile(0.05)
    vol_high = ranked["annualized_volatility"].quantile(0.95)
    ranked["volatility_score"] = normalize_series(
        ranked["annualized_volatility"],
        clip_low=vol_low,
        clip_high=vol_high,
    )

    ranked["momentum_persistence_score"] = normalize_series(
        ranked["momentum_persistence"],
        clip_low=0.0,
        clip_high=1.0,
    )

    ranked["liquidity_score"] = normalize_series(np.log1p(ranked["avg_dollar_volume"]))

    s2n_low = ranked["signal_to_noise"].quantile(0.05)
    s2n_high = ranked["signal_to_noise"].quantile(0.95)
    ranked["signal_to_noise_score"] = normalize_series(
        ranked["signal_to_noise"],
        clip_low=s2n_low,
        clip_high=s2n_high,
    )

    tq_low = ranked["trend_quality"].quantile(0.05)
    tq_high = ranked["trend_quality"].quantile(0.95)
    ranked["trend_quality_score"] = normalize_series(
        ranked["trend_quality"],
        clip_low=tq_low,
        clip_high=tq_high,
    )

    factor_cols = list(FACTOR_WEIGHTS.keys())
    ranked[factor_cols] = ranked[factor_cols].fillna(0.0)

    ranked["score"] = 0.0
    for col, weight in FACTOR_WEIGHTS.items():
        ranked["score"] += weight * ranked[col]

    ranked = ranked.sort_values(
        ["avg_dollar_volume", "score", "corr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


# =========================
# MAIN
# =========================
def main() -> None:
    print("===== TQQQ / QQQ NASDAQ UNIVERSE SCORING ANALYSIS =====")

    symbols = get_nasdaq_constituents()
    prices, volumes, failed = download_prices(symbols)

    print(f"[debug] price columns: {len(prices.columns)}")
    print(f"[debug] volume columns: {len(volumes.columns)}")
    print(f"[debug] contains {TARGET}: {TARGET in prices.columns}")
    print(f"[debug] contains {BENCHMARK}: {BENCHMARK in prices.columns}")

    prices.to_csv(PRICES_CSV)
    volumes.to_csv(VOLUMES_CSV)

    print("[compute] running analysis...")
    ranked = compute_metrics(prices, volumes)
    ranked.to_csv(OUT_CSV, index=False)

    print("\n===== TOP 30 (LOW-PRICE, HIGH-LIQUIDITY TQQQ/QQQ NASDAQ UNIVERSE CANDIDATES) =====")
    top30 = ranked.head(30)

    print(
        top30[
            [
                "rank",
                "symbol",
                "price",
                "avg_volume",
                "avg_dollar_volume",
                "same_direction_pct",
                "up_up_pct",
                "corr",
                "beta",
                "annualized_volatility",
                "momentum_persistence",
                "signal_to_noise",
                "trend_quality",
                "score",
            ]
        ].to_string(index=False)
    )

    print("\n===== TOP 10 CLOSEST MOVERS TO TQQQ =====")
    print(
        ranked.head(10)[
            [
                "rank",
                "symbol",
                "corr",
                "same_direction_pct",
                "beta",
                "score",
                "avg_dollar_volume",
            ]
        ].to_string(index=False)
    )

    top_symbols = top30["symbol"].tolist()

    print("\n[TOP 30 SYMBOLS]")
    print(top_symbols)

    print("\n[PYTHON LIST]")
    print(repr(top_symbols))

    with open(TOP30_TXT, "w", encoding="utf-8") as f:
        f.write(repr(top_symbols))

    print(f"\nSaved prices: {PRICES_CSV}")
    print(f"Saved volumes: {VOLUMES_CSV}")
    print(f"Saved ranking: {OUT_CSV}")
    print(f"Saved symbol list: {TOP30_TXT}")

    if failed:
        print(f"\n[warning] failed symbols: {len(failed)}")
        print(", ".join(failed[:25]))

    print("\n===== DONE =====")


if __name__ == "__main__":
    main()