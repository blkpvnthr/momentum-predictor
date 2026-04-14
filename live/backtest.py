from __future__ import annotations

import os
from datetime import time as dt_time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from trading_system import LiveTrader


THIS_FILE = Path(__file__).resolve()
LIVE_DIR = THIS_FILE.parent
PROJECT_ROOT = LIVE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

TIMEZONE = "America/New_York"
SYMBOLS = ["QQQ", "TQQQ", "SQQQ"]
INITIAL_EQUITY = 100_000.0
LOOKBACK_DAYS = 90


def load_alpaca_client() -> StockHistoricalDataClient:
    load_dotenv(ENV_PATH)

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    print(f"[backtest] loading env from: {ENV_PATH}")
    print(f"[backtest] .env exists: {ENV_PATH.exists()}")
    print(f"[backtest] APCA_API_KEY_ID present: {bool(api_key)}")
    print(f"[backtest] APCA_API_SECRET_KEY present: {bool(secret_key)}")

    if not api_key or not secret_key:
        raise RuntimeError(
            f"Missing Alpaca credentials in {ENV_PATH}. "
            "Expected APCA_API_KEY_ID and APCA_API_SECRET_KEY."
        )

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def _normalize_bar_frame(bars: pd.DataFrame) -> pd.DataFrame:
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


def download_last_3_months(
    symbols: list[str] = SYMBOLS,
    lookback_days: int = LOOKBACK_DAYS,
    feed: str = "iex",
) -> pd.DataFrame:
    client = load_alpaca_client()

    # Avoid recent SIP restriction and avoid deprecated utcnow usage.
    end_ts = pd.Timestamp.now("UTC") - pd.Timedelta(minutes=20)
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start_ts,
        end=end_ts,
        feed=feed,
    )

    print(
        f"[backtest] requesting Alpaca bars | symbols={symbols} | "
        f"start={start_ts} | end={end_ts} | timeframe=1Min | feed={feed}"
    )

    try:
        bars = client.get_stock_bars(request).df
    except APIError as e:
        raise RuntimeError(
            "Failed to download Alpaca bars. "
            "If you are on a free/basic plan, use feed='iex' or make sure "
            "the request end time is at least 15 minutes old."
        ) from e
    except Exception as e:
        raise RuntimeError("Unexpected Alpaca download failure.") from e

    if bars is None or len(bars) == 0:
        raise RuntimeError("No Alpaca bars returned.")

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

    merged["open"] = merged["qqq_open"]
    merged["high"] = merged["qqq_high"]
    merged["low"] = merged["qqq_low"]
    merged["close"] = merged["qqq_close"]
    merged["volume"] = merged["qqq_volume"]

    print(f"[backtest] merged rows={len(merged):,}")
    return merged


def run_backtest() -> pd.DataFrame:
    raw = download_last_3_months()

    trader = LiveTrader()
    feat = trader.build_features(raw)
    trader._load_vec_norm(feat)
    trader._reset_portfolio()

    print(f"[backtest] feature rows={len(feat):,}")
    print("[backtest] starting simulation...")

    records: list[dict[str, object]] = []

    for _, row in feat.iterrows():
        trader.update_position_state(row)
        obs = trader.get_obs(row)
        action = trader.select_action(obs)
        action_label = trader.execute_action(action, row)

        trader.update_position_state(row)
        trader.update_equity()

        records.append(
            {
                "timestamp": row["timestamp"],
                "qqq_close": float(row["qqq_close"]),
                "tqqq_close": float(row["tqqq_close"]),
                "sqqq_close": float(row["sqqq_close"]),
                "action": int(action),
                "action_label": action_label,
                "position_symbol": int(trader.position_symbol),
                "unrealized_pnl": float(trader.unrealized_pnl),
                "realized_pnl": float(trader.realized_pnl),
                "equity_norm": float(trader.equity),
                "regime": str(row["regime"]),
                "regime_conf": float(row["regime_conf"]),
                "signal_confidence": float(row["signal_confidence"]),
                "bull_score": float(row["bull_score"]),
                "bear_score": float(row["bear_score"]),
            }
        )

    results = pd.DataFrame(records)
    if results.empty:
        raise RuntimeError("Backtest produced no records.")

    results["strategy_equity"] = INITIAL_EQUITY * results["equity_norm"]

    qqq_start = float(results["qqq_close"].iloc[0])
    results["qqq_equity"] = INITIAL_EQUITY * (
        results["qqq_close"] / max(qqq_start, 1e-8)
    )

    print("[backtest] finished.")
    print(
        f"[backtest] final strategy equity={results['strategy_equity'].iloc[-1]:,.2f} | "
        f"final qqq equity={results['qqq_equity'].iloc[-1]:,.2f}"
    )

    return results


def plot_results(results: pd.DataFrame) -> None:
    results = results.copy()
    results["timestamp"] = pd.to_datetime(results["timestamp"], errors="coerce")
    results = results.dropna(subset=["timestamp"]).reset_index(drop=True)

    for col in ["qqq_close", "tqqq_close", "sqqq_close", "strategy_equity", "qqq_equity"]:
        results[col] = pd.to_numeric(results[col], errors="coerce")

    results = results.dropna(subset=["qqq_close", "strategy_equity", "qqq_equity"]).reset_index(drop=True)
    if results.empty:
        print("[backtest] no valid data to plot.")
        return

    strategy_start = float(results["strategy_equity"].iloc[0])
    results["strategy_equity_norm"] = results["strategy_equity"] / max(strategy_start, 1e-8)

    tqqq_buy_idx = results["action_label"].isin(["ENTER_TQQQ", "ENTER_TQQQ_TRANSITION"])
    tqqq_sell_idx = results["action_label"].isin(["EXIT_TQQQ", "FORCED_EXIT_TQQQ"])
    sqqq_buy_idx = results["action_label"].isin(["ENTER_SQQQ", "ENTER_SQQQ_TRANSITION"])
    sqqq_sell_idx = results["action_label"].isin(["EXIT_SQQQ", "FORCED_EXIT_SQQQ"])

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    # Top: QQQ price + normalized strategy equity overlay
    ax_price = axes[0]
    ax_price.plot(results["timestamp"], results["qqq_close"], linewidth=1.5, label="QQQ Price")
    ax_price.set_title("QQQ Price Action with Simulated Equity Overlay")
    ax_price.grid(alpha=0.3)

    ax_eq_overlay = ax_price.twinx()
    ax_eq_overlay.plot(
        results["timestamp"],
        results["strategy_equity_norm"],
        linewidth=1.5,
        alpha=0.85,
        label="Strategy Equity (normalized)",
        color="limegreen",
    )

    lines1, labels1 = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_eq_overlay.get_legend_handles_labels()
    ax_price.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Middle: equity vs equity
    axes[1].plot(results["timestamp"], results["strategy_equity"], linewidth=2, label="Strategy Equity")
    axes[1].plot(results["timestamp"], results["qqq_equity"], linewidth=2, label="QQQ Equity")
    axes[1].set_title("Simulated Strategy Equity vs QQQ Equity")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    # Bottom: actions on QQQ
    axes[2].plot(results["timestamp"], results["qqq_close"], linewidth=1.0, alpha=0.55, label="QQQ Price")
    axes[2].scatter(results.loc[tqqq_buy_idx, "timestamp"], results.loc[tqqq_buy_idx, "qqq_close"], s=20, label="TQQQ Entry")
    axes[2].scatter(results.loc[tqqq_sell_idx, "timestamp"], results.loc[tqqq_sell_idx, "qqq_close"], s=20, label="TQQQ Exit")
    axes[2].scatter(results.loc[sqqq_buy_idx, "timestamp"], results.loc[sqqq_buy_idx, "qqq_close"], s=20, label="SQQQ Entry")
    axes[2].scatter(results.loc[sqqq_sell_idx, "timestamp"], results.loc[sqqq_sell_idx, "qqq_close"], s=20, label="SQQQ Exit")
    axes[2].set_title("Trade Actions on QQQ Price")
    axes[2].legend(loc="upper left", ncol=2)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_backtest()
    plot_results(results)