from __future__ import annotations

import os
from datetime import time as dt_time
from pathlib import Path

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trading_env import TradingEnv
from trading_system import REGIME_ENGINE_PATH, build_features


THIS_FILE = Path(__file__).resolve()
LIVE_DIR = THIS_FILE.parent
PROJECT_ROOT = LIVE_DIR.parent

ENV_PATH = PROJECT_ROOT / ".env"
TRAINING_DATA_PATH = LIVE_DIR / "training_data.csv"
MODEL_OUT = LIVE_DIR / "trading_model"
VECNORM_OUT = LIVE_DIR / "vec_normalize.pkl"
TENSORBOARD_DIR = LIVE_DIR / "logs" / "tensorboard"

SYMBOLS = ["QQQ", "TQQQ", "SQQQ"]
START = "2024-01-01"
END = "2026-04-01"
TIMEZONE = "America/New_York"
TOTAL_TIMESTEPS = 300_000


def load_alpaca_client() -> StockHistoricalDataClient:
    """
    Load Alpaca credentials from PROJECT_ROOT/.env and build a historical data client.
    """
    load_dotenv(ENV_PATH)

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    print(f"[train] loading env from: {ENV_PATH}")
    print(f"[train] .env exists: {ENV_PATH.exists()}")
    print(f"[train] APCA_API_KEY_ID present: {bool(api_key)}")
    print(f"[train] APCA_API_SECRET_KEY present: {bool(secret_key)}")

    if not api_key or not secret_key:
        raise RuntimeError(
            f"Missing Alpaca credentials in {ENV_PATH}. "
            "Expected APCA_API_KEY_ID and APCA_API_SECRET_KEY."
        )

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def _normalize_bar_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Alpaca bar output into a standard flat DataFrame.
    """
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index()

    rename_map: dict[str, str] = {}
    if "timestamp" not in bars.columns and "time" in bars.columns:
        rename_map["time"] = "timestamp"
    if "open" not in bars.columns and "o" in bars.columns:
        rename_map["o"] = "open"
    if "high" not in bars.columns and "h" in bars.columns:
        rename_map["h"] = "high"
    if "low" not in bars.columns and "l" in bars.columns:
        rename_map["l"] = "low"
    if "close" not in bars.columns and "c" in bars.columns:
        rename_map["c"] = "close"
    if "volume" not in bars.columns and "v" in bars.columns:
        rename_map["v"] = "volume"

    if rename_map:
        bars = bars.rename(columns=rename_map)

    required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise RuntimeError(f"Downloaded bars missing required columns: {missing}")

    bars["timestamp"] = (
        pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
        .dt.tz_convert(TIMEZONE)
    )

    bars = (
        bars.dropna(subset=["timestamp"])
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    bars = bars[
        (bars["timestamp"].dt.time >= dt_time(9, 30))
        & (bars["timestamp"].dt.time <= dt_time(16, 0))
    ].copy()

    if len(bars) == 0:
        raise RuntimeError("No regular-hours data remained after filtering.")

    return bars


def download_training_data(
    symbols: list[str] = SYMBOLS,
    start: str = START,
    end: str = END,
    output_path: Path = TRAINING_DATA_PATH,
) -> pd.DataFrame:
    """
    Download synchronized QQQ/TQQQ/SQQQ minute bars.

    QQQ is the signal anchor.
    TQQQ and SQQQ are execution instruments.
    """
    client = load_alpaca_client()

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
    )

    print(
        f"[train] requesting Alpaca bars | symbols={symbols} | "
        f"start={start} | end={end} | timeframe=1Min"
    )

    try:
        bars = client.get_stock_bars(request).df
    except APIError as e:
        raise RuntimeError(
            "Failed to download Alpaca bars. "
            "Alpaca returned an authorization or data-access error. "
            "Check that PROJECT_ROOT/.env exists, that APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY are correct, and that the keys have access "
            "to the requested stock data endpoint."
        ) from e
    except Exception as e:
        raise RuntimeError(
            "Failed to download Alpaca bars for an unexpected reason."
        ) from e

    if bars is None or len(bars) == 0:
        raise RuntimeError(
            f"No Alpaca data returned for symbols={symbols}, start={start}, end={end}"
        )

    bars = _normalize_bar_frame(bars)

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        sdf = bars[bars["symbol"] == symbol].copy()
        if len(sdf) == 0:
            raise RuntimeError(f"No rows returned for symbol {symbol}")

        renamed = sdf.rename(
            columns={
                "open": f"{symbol.lower()}_open",
                "high": f"{symbol.lower()}_high",
                "low": f"{symbol.lower()}_low",
                "close": f"{symbol.lower()}_close",
                "volume": f"{symbol.lower()}_volume",
            }
        )[
            [
                "timestamp",
                f"{symbol.lower()}_open",
                f"{symbol.lower()}_high",
                f"{symbol.lower()}_low",
                f"{symbol.lower()}_close",
                f"{symbol.lower()}_volume",
            ]
        ]
        frames.append(renamed)

    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="timestamp", how="inner")

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # QQQ is the reference instrument for feature engineering.
    merged["open"] = merged["qqq_open"]
    merged["high"] = merged["qqq_high"]
    merged["low"] = merged["qqq_low"]
    merged["close"] = merged["qqq_close"]
    merged["volume"] = merged["qqq_volume"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"[train] saved {len(merged):,} merged rows to {output_path}")
    return merged


def make_env(df: pd.DataFrame):
    """
    Build the revised environment:
    - BULL -> only TQQQ entries
    - BEAR -> only SQQQ entries
    - QQQ drives signal generation
    """
    def _init():
        return TradingEnv(
            data=df,
            min_hold_bars=10,
            transaction_cost=0.0015,
            slippage_cost=0.0010,
            churn_penalty=0.0020,
            invalid_action_penalty=0.01,
            regime_violation_penalty=0.05,
            low_confidence_entry_penalty=0.01,
            min_signal_confidence=0.60,
            flat_reward=0.0002,
            holding_penalty=0.00005,
            transition_holding_penalty=0.0025,
            tqqq_extra_penalty=0.0010,
            sma_gate_penalty=0.01,
            tqqq_bull_strength_threshold=0.75,
            sqqq_bear_strength_threshold=0.75,
            adx_threshold=18.0,
            max_episode_steps=120,
            rolling_sharpe_window=20,
            sharpe_weight=0.10,
            downside_penalty_weight=0.15,
            drawdown_penalty_weight=0.15,
        )

    return _init


def main() -> None:
    print("[train] starting training process...")
    print("[train] downloading and preparing training data...")

    raw = download_training_data(
        symbols=SYMBOLS,
        start=START,
        end=END,
        output_path=TRAINING_DATA_PATH,
    )

    print(f"[train] raw rows={len(raw):,}")
    print("[train] building features and fitting regime engine...")

    df, regime_engine = build_features(
        raw,
        regime_engine=None,
        fit_regime_engine=True,
    )

    if regime_engine is None:
        raise RuntimeError("Failed to fit regime engine.")

    if len(df) == 0:
        raise RuntimeError("No rows remained after feature engineering.")

    print(f"[train] feature rows={len(df):,}")
    print("[train] regime distribution:")
    print(df["regime"].value_counts(dropna=False).to_string())

    REGIME_ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    regime_engine.save(REGIME_ENGINE_PATH)

    print("[train] building vectorized environment...")
    env = DummyVecEnv([make_env(df)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    print("[train] creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(TENSORBOARD_DIR),
    )

    print(f"[train] learning started | total_timesteps={TOTAL_TIMESTEPS:,}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    print("[train] saving model artifacts...")
    model.save(str(MODEL_OUT))
    env.save(str(VECNORM_OUT))

    print(f"[train] saved model to {MODEL_OUT}")
    print(f"[train] saved VecNormalize stats to {VECNORM_OUT}")
    print(f"[train] saved regime engine to {REGIME_ENGINE_PATH}")


if __name__ == "__main__":
    main()