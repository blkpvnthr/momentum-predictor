from __future__ import annotations

import os
from datetime import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# =========================================================
# ENV / PATHS
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# =========================================================
# CONFIG
# =========================================================
TIMEZONE = "America/New_York"

MONITOR_SYMBOLS = [
    "SPY",
    "SPXU",
    "DIA",
    "DOG",
    "QQQ",
    "SQQQ",
    "TQQQ",
    "VIXY",
    "IWM",
]

BULL_THRESHOLD = 0.15
BEAR_THRESHOLD = 0.15
TRANSITION_GAP_THRESHOLD = 0.10
MIN_CONFIDENCE = 0.20
FAILED_BREAKOUT_BUFFER = 0.0005
REVERSAL_CONFIRM_COUNT = 3

REGIME_MAP = {
    "BEAR": -1.0,
    "TRANSITION": 0.0,
    "BULL": 1.0,
}

UNIVERSE_MAP = {
    "INVERSE_ETF": -1.0,
    "NONE": 0.0,
    "NORMAL": 1.0,
}


# =========================================================
# ALPACA
# =========================================================
def get_client() -> StockHistoricalDataClient:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your .env."
        )

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def fetch_monitor_bars(
    start: str,
    end: str,
    symbols: List[str] | None = None,
) -> pd.DataFrame:
    if symbols is None:
        symbols = MONITOR_SYMBOLS

    client = get_client()

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
    )

    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise RuntimeError("No monitor bar data returned from Alpaca.")

    bars = bars.reset_index()

    rename_map = {}
    if "time" in bars.columns and "timestamp" not in bars.columns:
        rename_map["time"] = "timestamp"
    if "o" in bars.columns and "open" not in bars.columns:
        rename_map["o"] = "open"
    if "h" in bars.columns and "high" not in bars.columns:
        rename_map["h"] = "high"
    if "l" in bars.columns and "low" not in bars.columns:
        rename_map["l"] = "low"
    if "c" in bars.columns and "close" not in bars.columns:
        rename_map["c"] = "close"
    if "v" in bars.columns and "volume" not in bars.columns:
        rename_map["v"] = "volume"

    if rename_map:
        bars = bars.rename(columns=rename_map)

    required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise RuntimeError(f"Missing required monitor bar columns: {missing}")

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(TIMEZONE)
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    bars = bars[
        (bars["timestamp"].dt.time >= time(9, 30))
        & (bars["timestamp"].dt.time <= time(16, 0))
    ].copy()

    return bars


# =========================================================
# HELPERS
# =========================================================
def _safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_zscore(series: pd.Series, window: int) -> pd.Series:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - ma) / (std + 1e-9)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    x = np.arange(window, dtype=float)

    for i in range(window - 1, len(values)):
        y = values[i - window + 1 : i + 1]
        if np.any(~np.isfinite(y)):
            continue
        slope = np.polyfit(x, y, 1)[0]
        out[i] = slope / (abs(y[-1]) + 1e-9)

    return pd.Series(out, index=series.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================
# SYMBOL FEATURES
# =========================================================
def compute_symbol_feature_frame(sdf: pd.DataFrame) -> pd.DataFrame:
    sdf = sdf.copy().sort_values("timestamp").reset_index(drop=True)

    sdf["log_return"] = np.log(sdf["close"]).diff()
    sdf["ret_5"] = _safe_pct_change(sdf["close"], 5)
    sdf["ret_15"] = _safe_pct_change(sdf["close"], 15)
    sdf["ret_30"] = _safe_pct_change(sdf["close"], 30)
    sdf["vol_20"] = _safe_std(sdf["log_return"], 20)
    sdf["slope_15"] = _rolling_slope(sdf["close"], 15)
    sdf["zscore_20"] = _safe_zscore(sdf["close"], 20)

    sdf["prev_high_30"] = sdf["high"].shift(1).rolling(30).max()
    sdf["prev_low_30"] = sdf["low"].shift(1).rolling(30).min()

    sdf["breakout_attempted_prev_high"] = (
        sdf["high"] >= sdf["prev_high_30"] * (1.0 + FAILED_BREAKOUT_BUFFER)
    ).astype(float)

    sdf["failed_breakout_prev_high"] = (
        (sdf["breakout_attempted_prev_high"] == 1.0)
        & (sdf["close"] < sdf["prev_high_30"])
        & (sdf["close"] < sdf["open"])
    ).astype(float)

    sdf["breakdown_attempted_prev_low"] = (
        sdf["low"] <= sdf["prev_low_30"] * (1.0 - FAILED_BREAKOUT_BUFFER)
    ).astype(float)

    sdf["failed_breakdown_prev_low"] = (
        (sdf["breakdown_attempted_prev_low"] == 1.0)
        & (sdf["close"] > sdf["prev_low_30"])
        & (sdf["close"] > sdf["open"])
    ).astype(float)

    sdf["directional_score"] = (
        0.40 * sdf["ret_5"]
        + 0.35 * sdf["ret_15"]
        + 0.25 * sdf["ret_30"]
        + 0.30 * sdf["slope_15"]
        + 0.10 * np.tanh(sdf["zscore_20"] / 3.0)
    )

    sdf["pressure_score"] = (
        0.50 * sdf["directional_score"]
        - 0.25 * sdf["failed_breakout_prev_high"]
        + 0.25 * sdf["failed_breakdown_prev_low"]
    )

    keep = [
        "timestamp",
        "symbol",
        "ret_5",
        "ret_15",
        "ret_30",
        "vol_20",
        "slope_15",
        "zscore_20",
        "breakout_attempted_prev_high",
        "failed_breakout_prev_high",
        "breakdown_attempted_prev_low",
        "failed_breakdown_prev_low",
        "directional_score",
        "pressure_score",
    ]
    return sdf[keep].copy()


def build_monitor_feature_matrix(
    bars: pd.DataFrame,
    symbols: List[str] | None = None,
) -> pd.DataFrame:
    if symbols is None:
        symbols = MONITOR_SYMBOLS

    frames = []
    for symbol in symbols:
        sdf = bars[bars["symbol"] == symbol].copy()
        if sdf.empty:
            continue

        f = compute_symbol_feature_frame(sdf)
        f = f.rename(
            columns={
                "ret_5": f"{symbol.lower()}_ret_5",
                "ret_15": f"{symbol.lower()}_ret_15",
                "ret_30": f"{symbol.lower()}_ret_30",
                "vol_20": f"{symbol.lower()}_vol_20",
                "slope_15": f"{symbol.lower()}_slope_15",
                "zscore_20": f"{symbol.lower()}_zscore_20",
                "breakout_attempted_prev_high": f"{symbol.lower()}_attempt_hi",
                "failed_breakout_prev_high": f"{symbol.lower()}_failed_hi",
                "breakdown_attempted_prev_low": f"{symbol.lower()}_attempt_lo",
                "failed_breakdown_prev_low": f"{symbol.lower()}_failed_lo",
                "directional_score": f"{symbol.lower()}_dir_score",
                "pressure_score": f"{symbol.lower()}_pressure_score",
            }
        )
        f = f.drop(columns=["symbol"])
        frames.append(f)

    if not frames:
        raise RuntimeError("No monitor feature frames were built.")

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="timestamp", how="outer")

    return out.sort_values("timestamp").reset_index(drop=True)


# =========================================================
# REGIME INFERENCE (VECTORIZED)
# =========================================================
def infer_regime_df(regime_df: pd.DataFrame) -> pd.DataFrame:
    df = regime_df.copy()

    spy = df["spy_dir_score"].fillna(0.0)
    dia = df["dia_dir_score"].fillna(0.0)
    qqq = df["qqq_dir_score"].fillna(0.0)
    tqqq = df["tqqq_dir_score"].fillna(0.0)
    iwm = df["iwm_dir_score"].fillna(0.0)

    spxu = df["spxu_dir_score"].fillna(0.0)
    dog = df["dog_dir_score"].fillna(0.0)
    sqqq = df["sqqq_dir_score"].fillna(0.0)
    vixy = df["vixy_dir_score"].fillna(0.0)

    spy_fail_hi = df.get("spy_failed_hi", pd.Series(0.0, index=df.index)).fillna(0.0)
    qqq_fail_hi = df.get("qqq_failed_hi", pd.Series(0.0, index=df.index)).fillna(0.0)
    dia_fail_hi = df.get("dia_failed_hi", pd.Series(0.0, index=df.index)).fillna(0.0)

    spy_fail_lo = df.get("spy_failed_lo", pd.Series(0.0, index=df.index)).fillna(0.0)
    qqq_fail_lo = df.get("qqq_failed_lo", pd.Series(0.0, index=df.index)).fillna(0.0)
    dia_fail_lo = df.get("dia_failed_lo", pd.Series(0.0, index=df.index)).fillna(0.0)

    bull_raw = (
        spy + dia + qqq + tqqq + iwm
        - spxu - dog - sqqq - vixy
    ) / 9.0

    bear_raw = (
        -spy - dia - qqq - tqqq - iwm
        + spxu + dog + sqqq + vixy
    ) / 9.0

    # failed upside breakout pressure hurts bull case
    bull_raw = bull_raw - 0.10 * ((spy_fail_hi + qqq_fail_hi + dia_fail_hi) / 3.0)

    # failed downside breakdown pressure hurts bear case
    bear_raw = bear_raw - 0.10 * ((spy_fail_lo + qqq_fail_lo + dia_fail_lo) / 3.0)

    bull_score = np.tanh(bull_raw * 12.0)
    bear_score = np.tanh(bear_raw * 12.0)

    score_gap = bull_score - bear_score
    abs_gap = np.abs(score_gap)
    clipped_gap = np.minimum(abs_gap, 1.0)
    transition_score = 1.0 - clipped_gap

    inferred_regime = np.where(
        (bull_score >= BULL_THRESHOLD) & (score_gap >= TRANSITION_GAP_THRESHOLD),
        "BULL",
        np.where(
            (bear_score >= BEAR_THRESHOLD) & (-score_gap >= TRANSITION_GAP_THRESHOLD),
            "BEAR",
            "TRANSITION",
        ),
    )

    regime_confidence = np.where(
        inferred_regime == "TRANSITION",
        1.0 - clipped_gap,
        clipped_gap,
    )

    regime_inferred = np.where(
        inferred_regime == "BULL",
        REGIME_MAP["BULL"],
        np.where(
            inferred_regime == "BEAR",
            REGIME_MAP["BEAR"],
            REGIME_MAP["TRANSITION"],
        ),
    ).astype(float)

    out = pd.DataFrame(
        {
            "inferred_regime": inferred_regime,
            "regime_inferred": regime_inferred,
            "regime_confidence": regime_confidence.astype(float),
            "bull_score": bull_score.astype(float),
            "bear_score": bear_score.astype(float),
            "transition_score": transition_score.astype(float),
            "score_gap": score_gap.astype(float),
        },
        index=df.index,
    )

    return out


# =========================================================
# HISTORICAL ACTIVE REGIME STATE MACHINE
# =========================================================
def build_active_regime_state(regime_df: pd.DataFrame) -> pd.DataFrame:
    df = regime_df.copy().reset_index(drop=True)

    active_regimes: list[str] = []
    candidate_regimes: list[str] = []
    candidate_counts: list[float] = []
    reversal_watches: list[float] = []
    flip_confirmed: list[float] = []
    selected_universe_num: list[float] = []
    trading_enabled: list[float] = []

    active_regime = "TRANSITION"
    candidate_regime = ""
    candidate_count = 0

    for _, row in df.iterrows():
        inferred_regime = row["inferred_regime"]
        confidence = float(row["regime_confidence"])

        bull_failed_high = (
            row.get("spy_failed_hi", 0.0)
            + row.get("qqq_failed_hi", 0.0)
            + row.get("dia_failed_hi", 0.0)
        ) > 0.0

        bear_failed_low = (
            row.get("spy_failed_lo", 0.0)
            + row.get("qqq_failed_lo", 0.0)
            + row.get("dia_failed_lo", 0.0)
        ) > 0.0

        reversal_watch = 0.0
        flip = 0.0
        desired_candidate = ""

        if active_regime == "BULL":
            if bull_failed_high:
                reversal_watch = 1.0
                desired_candidate = "TRANSITION"
                if inferred_regime == "BEAR" and confidence >= MIN_CONFIDENCE:
                    desired_candidate = "BEAR"

        elif active_regime == "BEAR":
            if bear_failed_low:
                reversal_watch = 1.0
                desired_candidate = "TRANSITION"
                if inferred_regime == "BULL" and confidence >= MIN_CONFIDENCE:
                    desired_candidate = "BULL"

        else:
            if inferred_regime in {"BULL", "BEAR"} and confidence >= MIN_CONFIDENCE:
                reversal_watch = 1.0
                desired_candidate = inferred_regime

        if desired_candidate == "":
            candidate_regime = ""
            candidate_count = 0
        else:
            if desired_candidate == candidate_regime:
                candidate_count += 1
            else:
                candidate_regime = desired_candidate
                candidate_count = 1

            if candidate_count >= REVERSAL_CONFIRM_COUNT:
                if active_regime != candidate_regime:
                    active_regime = candidate_regime
                    flip = 1.0
                candidate_regime = ""
                candidate_count = 0
                reversal_watch = 0.0

        if active_regime == "BULL" and confidence >= MIN_CONFIDENCE:
            trading_flag = 1.0
            universe_num = UNIVERSE_MAP["NORMAL"]
        elif active_regime == "BEAR" and confidence >= MIN_CONFIDENCE:
            trading_flag = 1.0
            universe_num = UNIVERSE_MAP["INVERSE_ETF"]
        else:
            trading_flag = 0.0
            universe_num = UNIVERSE_MAP["NONE"]

        active_regimes.append(active_regime)
        candidate_regimes.append(candidate_regime)
        candidate_counts.append(float(candidate_count))
        reversal_watches.append(float(reversal_watch))
        flip_confirmed.append(float(flip))
        selected_universe_num.append(float(universe_num))
        trading_enabled.append(float(trading_flag))

    df["active_regime_str"] = active_regimes
    df["candidate_regime"] = candidate_regimes
    df["candidate_count"] = candidate_counts
    df["reversal_watch"] = reversal_watches
    df["flip_confirmed"] = flip_confirmed
    df["selected_universe_num"] = selected_universe_num
    df["trading_enabled"] = trading_enabled
    df["regime_active"] = df["active_regime_str"].map(REGIME_MAP).astype(float)

    return df


# =========================================================
# MAIN BUILDER
# =========================================================
def build_historical_regime_series(
    start: str,
    end: str,
) -> pd.DataFrame:
    bars = fetch_monitor_bars(start=start, end=end)
    regime_df = build_monitor_feature_matrix(bars)

    inferred = infer_regime_df(regime_df)
    regime_df = pd.concat([regime_df, inferred], axis=1)

    regime_df = build_active_regime_state(regime_df)

    # ensure downstream-required columns exist even if sparse
    for col in [
        "reversal_watch",
        "candidate_count",
        "flip_confirmed",
        "trading_enabled",
        "selected_universe_num",
        "spy_ret_5",
        "qqq_ret_5",
        "vixy_ret_5",
    ]:
        if col not in regime_df.columns:
            regime_df[col] = 0.0

    return regime_df.sort_values("timestamp").reset_index(drop=True)