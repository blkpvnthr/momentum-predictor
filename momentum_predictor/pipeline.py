from __future__ import annotations

import os
import time
from datetime import time as dt_time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from momentum_predictor.config import SEQ_LEN
from momentum_predictor.historical_regime import build_historical_regime_series


# =========================================================
# ENV / PATHS
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# =========================================================
# PIPELINE CONFIG
# =========================================================
CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USE_FEATURE_CLIPPING = True
CLIP_VALUE = 5.0

# Keep targets on real scale.
USE_TARGET_TRANSFORM = False
TARGET_SCALE = 10.0  # only used if USE_TARGET_TRANSFORM=True

USE_QUANTUM_FEATURES = True

WARMUP_BARS = 200
MAX_HORIZON = 30
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)


# =========================================================
# MEMORY HELPERS
# =========================================================

def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def normalize_df(
    df: pd.DataFrame,
    cols: List[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])

    if USE_FEATURE_CLIPPING:
        df[cols] = df[cols].clip(-CLIP_VALUE, CLIP_VALUE)

    return df, scaler


def require_columns(df: pd.DataFrame, required: List[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


# =========================================================
# DATA LOADING
# =========================================================
def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Expected APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env."
        )

    client = StockHistoricalDataClient(
        api_key=api_key,
        secret_key=secret_key,
    )

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    if df is None or len(df) == 0:
        raise RuntimeError(f"No market data returned for symbol={symbol}, start={start}, end={end}")

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    rename_map = {}
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()

    if "timestamp" not in df.columns and "time" in df.columns:
        rename_map["time"] = "timestamp"
    if "open" not in df.columns and "o" in df.columns:
        rename_map["o"] = "open"
    if "high" not in df.columns and "h" in df.columns:
        rename_map["h"] = "high"
    if "low" not in df.columns and "l" in df.columns:
        rename_map["l"] = "low"
    if "close" not in df.columns and "c" in df.columns:
        rename_map["c"] = "close"
    if "volume" not in df.columns and "v" in df.columns:
        rename_map["v"] = "volume"

    if rename_map:
        df = df.rename(columns=rename_map)

    require_columns(df, ["timestamp", "open", "high", "low", "close", "volume"], "alpaca bars")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert("America/New_York")
    df = df.dropna(subset=["timestamp"]).copy()

    df = df[
        (df["timestamp"].dt.time >= MARKET_OPEN)
        & (df["timestamp"].dt.time <= MARKET_CLOSE)
    ].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No regular-hours bars remained after market-hours filter.")

    return df


# =========================================================
# 5M RESAMPLE
# =========================================================
def build_5m_bars(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["timestamp", "open", "high", "low", "close", "volume"], "1m bars")

    df = df.copy().set_index("timestamp")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_5m = (
        df.resample("5min", label="right", closed="right")
        .agg(agg)
        .dropna()
    )

    return df_5m.reset_index()


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["hour"] = df["timestamp"].dt.hour.astype(np.float32)
    df["minute"] = df["timestamp"].dt.minute.astype(np.float32)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(np.float32)

    df["minutes_from_open"] = (minutes - (9 * 60 + 30)).astype(np.float32)
    df["minutes_to_close"] = ((16 * 60) - minutes).astype(np.float32)

    df["is_opening_window"] = (df["minutes_from_open"] <= 30).astype(np.float32)
    df["is_midday"] = (
        (df["minutes_from_open"] > 90) & (df["minutes_to_close"] > 120)
    ).astype(np.float32)
    df["is_power_hour"] = (df["minutes_to_close"] <= 60).astype(np.float32)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    require_columns(df, ["timestamp", "open", "high", "low", "close", "volume"], "feature input")

    df = add_time_features(df)

    # Returns / bar structure
    df["log_return"] = np.log(df["close"]).diff()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["ret_15"] = df["close"].pct_change(15)

    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]

    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Volatility / momentum
    df["volatility_20"] = df["log_return"].rolling(20).std()
    df["vol_30"] = df["ret_1"].rolling(30).std()
    df["vol_60"] = df["ret_1"].rolling(60).std()

    for w in (3, 5, 10, 20, 30, 60):
        df[f"momentum_{w}"] = df["close"].pct_change(w)

    # Trend
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_15"] = df["close"].rolling(15).mean()
    df["sma_30"] = df["close"].rolling(30).mean()
    df["sma_60"] = df["close"].rolling(60).mean()

    df["dist_sma_5"] = df["close"] / (df["sma_5"] + 1e-6) - 1.0
    df["dist_sma_15"] = df["close"] / (df["sma_15"] + 1e-6) - 1.0
    df["dist_sma_30"] = df["close"] / (df["sma_30"] + 1e-6) - 1.0
    df["dist_sma_60"] = df["close"] / (df["sma_60"] + 1e-6) - 1.0

    df["slope_sma_5"] = df["sma_5"].pct_change(3)
    df["slope_sma_15"] = df["sma_15"].pct_change(3)
    df["slope_sma_30"] = df["sma_30"].pct_change(3)

    # VWAP
    df["date"] = df["timestamp"].dt.date
    price = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]

    df["vwap"] = (
        (price * vol).groupby(df["date"]).cumsum()
        / (vol.groupby(df["date"]).cumsum() + 1e-6)
    )

    df["vwap_dist"] = df["close"] - df["vwap"]
    df["vwap_dist_norm"] = df["vwap_dist"] / (df["volatility_20"] + 1e-6)
    df["vwap_slope"] = df["vwap"].diff()

    # Volume
    df["volume_mean_20"] = df["volume"].rolling(20).mean()
    df["volume_std_20"] = df["volume"].rolling(20).std()
    df["volume_z"] = (
        (df["volume"] - df["volume_mean_20"])
        / (df["volume_std_20"] + 1e-6)
    )
    df["rel_volume"] = df["volume"] / (df["volume_mean_20"] + 1e-6)

    # Flow / pressure
    df["clv"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-6)
    df["signed_volume"] = np.sign(df["log_return"].fillna(0.0)) * df["volume"]

    df["pressure"] = df["body"] * df["volume"]
    df["expansion"] = df["range"] * df["volume"]

    # Breakout / range structure
    df["rolling_high"] = df["high"].rolling(60).max()
    df["rolling_low"] = df["low"].rolling(60).min()

    df["dist_to_high"] = (
        (df["close"] - df["rolling_high"])
        / (df["volatility_20"] + 1e-6)
    )
    df["dist_to_low"] = (
        (df["close"] - df["rolling_low"])
        / (df["volatility_20"] + 1e-6)
    )

    df["rolling_high_15"] = df["high"].rolling(15).max()
    df["rolling_low_15"] = df["low"].rolling(15).min()

    df["dist_high_15"] = df["close"] / (df["rolling_high_15"] + 1e-6) - 1.0
    df["dist_low_15"] = df["close"] / (df["rolling_low_15"] + 1e-6) - 1.0
    df["pos_in_range_15"] = (
        (df["close"] - df["rolling_low_15"])
        / ((df["rolling_high_15"] - df["rolling_low_15"]) + 1e-6)
    )

    # Regime helpers
    df["accel_1_5"] = df["ret_1"] - df["ret_5"]
    df["accel_3_15"] = df["ret_3"] - df["ret_15"]
    df["vol_ratio_15_60"] = df["vol_30"] / (df["vol_60"] + 1e-6)

    # Simple RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Execution helpers
    df["trend_ok"] = (
        (df["ret_5"] > 0)
        & (df["slope_sma_15"] > 0)
        & (df["dist_sma_15"] > -0.002)
    ).astype(np.float32)

    df["breakout_ok"] = (df["dist_high_15"] >= -0.0025).astype(np.float32)

    return df


# =========================================================
# OPTIONAL QUANTUM FEATURES
# =========================================================
def attach_quantum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional hook. This only runs if momentum_predictor.quantum_features exists
    and USE_QUANTUM_FEATURES is enabled.
    """
    if not USE_QUANTUM_FEATURES:
        return df

    try:
        from momentum_predictor.quantum_features import QuantumFeatureConfig, QuantumFeatureModule
    except Exception as exc:
        print(f"[pipeline] quantum features unavailable, skipping: {exc}")
        return df

    quantum_feature_inputs = [
        c for c in [
            "vol_30",
            "vol_60",
            "dist_high_15",
            "dist_sma_30",
            "rsi_14",
            "ret_5",
            "slope_sma_15",
            "minutes_to_close",
        ]
        if c in df.columns
    ]

    if len(quantum_feature_inputs) < 4:
        print("[pipeline] not enough inputs for quantum features, skipping")
        return df

    qcfg = QuantumFeatureConfig(
        feature_cols=quantum_feature_inputs,
        n_qubits=6,
        n_layers=2,
        clip_value=3.0,
        random_state=42,
    )
    module = QuantumFeatureModule(qcfg).fit(df)
    q_df = module.transform(df)

    df = pd.concat([df.reset_index(drop=True), q_df.reset_index(drop=True)], axis=1)
    return df


# =========================================================
# REGIME CACHE
# =========================================================
def _cache_key_for_regime(start: str, end: str) -> str:
    safe_start = start.replace(":", "").replace("+", "").replace("-", "")
    safe_end = end.replace(":", "").replace("+", "").replace("-", "")
    return f"historical_regime_{safe_start}_{safe_end}.csv"


def load_or_build_regime_df(start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    cache_path = CACHE_DIR / _cache_key_for_regime(start, end)

    if use_cache and cache_path.exists():
        t0 = time.time()
        regime_df = pd.read_csv(cache_path)
        regime_df["timestamp"] = pd.to_datetime(
            regime_df["timestamp"], utc=True, errors="coerce"
        ).dt.tz_convert("America/New_York")
        regime_df = regime_df.dropna(subset=["timestamp"]).copy()
        print(
            f"[pipeline] loaded cached regime data: {cache_path.name} "
            f"in {time.time() - t0:.2f}s"
        )
        return regime_df

    t0 = time.time()
    regime_df = build_historical_regime_series(start=start, end=end)
    regime_df["timestamp"] = pd.to_datetime(
        regime_df["timestamp"], utc=True, errors="coerce"
    ).dt.tz_convert("America/New_York")
    regime_df = regime_df.dropna(subset=["timestamp"]).copy()
    build_secs = time.time() - t0

    if use_cache:
        regime_df.to_csv(cache_path, index=False)
        print(
            f"[pipeline] built and cached regime data: {cache_path.name} "
            f"in {build_secs:.2f}s"
        )
    else:
        print(f"[pipeline] built regime data in {build_secs:.2f}s")

    return regime_df


# =========================================================
# REGIME FEATURES
# =========================================================
def attach_regime_features(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = df_1m["timestamp"].min().tz_convert("UTC").isoformat()
    end = df_1m["timestamp"].max().tz_convert("UTC").isoformat()

    regime_df = load_or_build_regime_df(start=start, end=end, use_cache=use_cache)

    regime_cols = [
        "timestamp",
        "inferred_regime",
        "active_regime_str",
        "regime_inferred",
        "regime_active",
        "regime_confidence",
        "bull_score",
        "bear_score",
        "transition_score",
        "score_gap",
        "reversal_watch",
        "candidate_count",
        "flip_confirmed",
        "trading_enabled",
        "selected_universe_num",
        "spy_ret_5",
        "qqq_ret_5",
        "vixy_ret_5",
    ]

    missing_cols = [c for c in regime_cols if c not in regime_df.columns]
    if missing_cols:
        raise RuntimeError(
            f"historical_regime.py is missing required columns: {missing_cols}"
        )

    regime_df = reduce_memory(regime_df[regime_cols].copy())

    df_1m = pd.merge_asof(
        df_1m.sort_values("timestamp"),
        regime_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    df_5m = pd.merge_asof(
        df_5m.sort_values("timestamp"),
        regime_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    return df_1m, df_5m


# =========================================================
# LABELS
# =========================================================
def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    require_columns(df, ["close", "high", "low", "volatility_20"], "label input")

    recent_high = df["high"].shift(1).rolling(20).max()
    recent_low = df["low"].shift(1).rolling(20).min()
    recent_range = (recent_high - recent_low).clip(lower=eps)

    short_dir = np.sign(df["close"].pct_change(5)).fillna(0.0)

    for h in (5, 15, 30):
        df[f"ret_{h}"] = np.log(df["close"].shift(-h) / df["close"])

        future_high = df["high"].shift(-1).rolling(h).max()
        future_low = df["low"].shift(-1).rolling(h).min()
        future_close = df["close"].shift(-h)

        up_excur = (future_high - df["close"]) / (df["close"] + eps)
        down_excur = (df["close"] - future_low) / (df["close"] + eps)

        breakout_up_level = recent_high + 0.10 * recent_range
        breakout_down_level = recent_low - 0.10 * recent_range

        broke_up = future_high > breakout_up_level
        broke_down = future_low < breakout_down_level

        vol = df["volatility_20"].fillna(0.0)
        min_move = (1.25 * vol).clip(lower=0.0005)

        realized_up = df[f"ret_{h}"] > min_move
        realized_down = df[f"ret_{h}"] < -min_move

        df[f"breakout_up_{h}"] = (broke_up & realized_up).astype(np.float32)
        df[f"breakout_down_{h}"] = (broke_down & realized_down).astype(np.float32)

        failed_up = (
            broke_up
            & (future_close < recent_high)
            & (df[f"ret_{h}"] <= 0)
        )

        failed_down = (
            broke_down
            & (future_close > recent_low)
            & (df[f"ret_{h}"] >= 0)
        )

        future_dir = np.sign(df[f"ret_{h}"]).fillna(0.0)

        continuation = (
            (short_dir != 0)
            & (future_dir == short_dir)
            & ~failed_up
            & ~failed_down
        )

        df[f"continuation_{h}"] = continuation.astype(np.float32)

        df[f"up_excur_{h}"] = up_excur
        df[f"down_excur_{h}"] = down_excur
        df[f"failed_breakout_up_{h}"] = failed_up.astype(np.float32)
        df[f"failed_breakout_down_{h}"] = failed_down.astype(np.float32)

    return df


# =========================================================
# ALIGN 5M -> 1M
# =========================================================
def align_5m_to_1m(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    df_5m = df_5m.copy()
    df_5m = df_5m.rename(columns=lambda c: f"{c}_5m" if c != "timestamp" else c)

    df = pd.merge_asof(
        df_1m.sort_values("timestamp"),
        df_5m.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    return df


# =========================================================
# SEQUENCES
# =========================================================
def make_sequences(
    df: pd.DataFrame,
    feature_cols_1m: List[str],
    feature_cols_5m: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_cols = [
        "ret_5",
        "ret_15",
        "ret_30",
        "breakout_up_15",
        "breakout_down_15",
        "continuation_15",
    ]

    require_columns(df, feature_cols_1m + feature_cols_5m + target_cols + ["timestamp"], "sequence input")

    df = df.dropna(subset=feature_cols_1m + feature_cols_5m + target_cols).reset_index(drop=True)

    X1, X5, y, timestamps = [], [], [], []

    for i in range(SEQ_LEN, len(df)):
        X1.append(df.loc[i - SEQ_LEN : i - 1, feature_cols_1m].values.astype(np.float32))
        X5.append(df.loc[i - SEQ_LEN : i - 1, feature_cols_5m].values.astype(np.float32))

        target = df.loc[i, target_cols].values.astype(np.float32)

        if USE_TARGET_TRANSFORM:
            target[:3] = np.tanh(target[:3] * TARGET_SCALE)

        y.append(target)
        timestamps.append(df.loc[i, "timestamp"])

    return (
        np.asarray(X1, dtype=np.float32),
        np.asarray(X5, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(timestamps),
    )


# =========================================================
# PIPELINE
# =========================================================
def run_pipeline(
    symbol: str = "QQQ",
    start: str = "2025-09-01",
    end: str = "2026-03-01",
    use_regime_cache: bool = True,
):
    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------
    df_1m = load_data(symbol, start, end)
    df_5m = build_5m_bars(df_1m)

    df_1m = reduce_memory(df_1m)
    df_5m = reduce_memory(df_5m)

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------
    df_1m = build_features(df_1m)
    df_5m = build_features(df_5m)

    # Optional quantum features
    df_1m = attach_quantum_features(df_1m)
    df_5m = attach_quantum_features(df_5m)

    # -----------------------------------------------------
    # REGIME FEATURES
    # -----------------------------------------------------
    t0 = time.time()
    df_1m, df_5m = attach_regime_features(
        df_1m,
        df_5m,
        use_cache=use_regime_cache,
    )

    df_1m = reduce_memory(df_1m)
    df_5m = reduce_memory(df_5m)

    print(f"[pipeline] regime attach time: {time.time() - t0:.2f}s")

    # -----------------------------------------------------
    # ALIGN 5M → 1M
    # -----------------------------------------------------
    df = align_5m_to_1m(df_1m, df_5m)

    del df_1m, df_5m
    df = reduce_memory(df)

    # -----------------------------------------------------
    # LABELS
    # -----------------------------------------------------
    df = build_labels(df)

    for col in ["inferred_regime", "active_regime_str", "date", "date_5m"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # -----------------------------------------------------
    # FEATURE SET
    # -----------------------------------------------------
    base_feature_cols = [
        "log_return",
        "ret_1",
        "ret_3",
        "ret_5",
        "ret_10",
        "ret_15",
        "body",
        "range",
        "upper_wick",
        "lower_wick",
        "volatility_20",
        "vol_30",
        "vol_60",
        "momentum_5",
        "momentum_10",
        "momentum_20",
        "momentum_30",
        "vwap_dist_norm",
        "vwap_slope",
        "volume_z",
        "rel_volume",
        "clv",
        "signed_volume",
        "pressure",
        "expansion",
        "dist_to_high",
        "dist_to_low",
        "dist_high_15",
        "dist_low_15",
        "pos_in_range_15",
        "dist_sma_5",
        "dist_sma_15",
        "dist_sma_30",
        "slope_sma_5",
        "slope_sma_15",
        "slope_sma_30",
        "accel_1_5",
        "accel_3_15",
        "vol_ratio_15_60",
        "rsi_14",
        "hour",
        "minute",
        "day_of_week",
        "minutes_from_open",
        "minutes_to_close",
        "is_opening_window",
        "is_midday",
        "is_power_hour",
        "trend_ok",
        "breakout_ok",
    ]

    if USE_QUANTUM_FEATURES:
        base_feature_cols += [
            "quantum_score",
            "quantum_energy",
            "quantum_dispersion",
            "quantum_alignment",
        ]

    regime_feature_cols = [
        "regime_inferred",
        "regime_active",
        "regime_confidence",
        "bull_score",
        "bear_score",
        "transition_score",
        "score_gap",
        "trading_enabled",
        "selected_universe_num",
        "spy_ret_5",
        "qqq_ret_5",
        "vixy_ret_5",
    ]

    feature_cols_1m = [c for c in base_feature_cols + regime_feature_cols if c in df.columns]
    feature_cols_5m = [f"{c}_5m" for c in feature_cols_1m if f"{c}_5m" in df.columns]

    require_columns(df, feature_cols_1m + feature_cols_5m, "post-align feature frame")

    # -----------------------------------------------------
    # TRIM (warmup + forward horizon)
    # -----------------------------------------------------
    if len(df) <= WARMUP_BARS + MAX_HORIZON:
        raise ValueError(
            f"Not enough rows after feature generation. "
            f"Need > {WARMUP_BARS + MAX_HORIZON}, got {len(df)}"
        )

    df = df.iloc[WARMUP_BARS:-MAX_HORIZON].reset_index(drop=True)

    # -----------------------------------------------------
    # BUILD SEQUENCES
    # -----------------------------------------------------
    X1, X5, y, timestamps = make_sequences(df, feature_cols_1m, feature_cols_5m)

    last_row = df.iloc[-1].copy()
    del df

    # -----------------------------------------------------
    # DEBUG SUMMARY
    # -----------------------------------------------------
    universe_num = float(last_row["selected_universe_num"])
    if universe_num > 0:
        selected_universe_str = "NORMAL"
    elif universe_num < 0:
        selected_universe_str = "INVERSE_ETF"
    else:
        selected_universe_str = "NONE"

    print(f"[pipeline] X1: {X1.shape}, X5: {X5.shape}, y: {y.shape}")
    print(
        "[pipeline] historical regime features attached "
        f"| last_inferred_num={last_row['regime_inferred']:.0f} "
        f"| last_active_num={last_row['regime_active']:.0f} "
        f"| last_confidence={last_row['regime_confidence']:.4f} "
        f"| last_trading_enabled={last_row['trading_enabled']:.0f} "
        f"| last_selected_universe_num={universe_num:.0f} "
        f"| last_selected_universe_str={selected_universe_str}"
    )

    return X1, X5, y, timestamps