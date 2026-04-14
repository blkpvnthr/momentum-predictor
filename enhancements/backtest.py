from __future__ import annotations

import math
import os
from datetime import time as dt_time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
BAR_TIMEFRAME = TimeFrame.Minute

# Baseline friction approximation
BASELINE_SWITCH_COST = 0.0015


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
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

    if bars.empty:
        raise RuntimeError("No regular-hours data remained after filtering.")

    return bars


def download_last_3_months(
    symbols: list[str] = SYMBOLS,
    lookback_days: int = LOOKBACK_DAYS,
    feed: str = "iex",
) -> pd.DataFrame:
    client = load_alpaca_client()

    end_ts = pd.Timestamp.now("UTC") - pd.Timedelta(minutes=20)
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=BAR_TIMEFRAME,
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
    except APIError as exc:
        raise RuntimeError(
            "Failed to download Alpaca bars. "
            "If you are on a free/basic plan, use feed='iex' and make sure "
            "the request end time is at least 15 minutes old."
        ) from exc
    except Exception as exc:
        raise RuntimeError("Unexpected Alpaca download failure.") from exc

    if bars is None or len(bars) == 0:
        raise RuntimeError("No Alpaca bars returned.")

    bars = _normalize_bar_frame(bars)

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        sdf = bars[bars["symbol"] == symbol].copy()
        if sdf.empty:
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

    # QQQ remains the signal anchor for feature engineering.
    merged["open"] = merged["qqq_open"]
    merged["high"] = merged["qqq_high"]
    merged["low"] = merged["qqq_low"]
    merged["close"] = merged["qqq_close"]
    merged["volume"] = merged["qqq_volume"]

    print(f"[backtest] merged rows={len(merged):,}")
    return merged


# ---------------------------------------------------------
# METRICS / BASELINES
# ---------------------------------------------------------
def _max_drawdown(equity: pd.Series) -> float:
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def _annualized_sharpe_from_bar_returns(returns: pd.Series, bars_per_year: int) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2:
        return float("nan")

    std = float(returns.std())
    if std <= 1e-12:
        return float("nan")

    mean = float(returns.mean())
    return math.sqrt(bars_per_year) * (mean / std)


def _summarize_equity(name: str, equity: pd.Series, bars_per_year: int) -> dict[str, float | str]:
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    if len(equity) < 2:
        return {
            "name": name,
            "final_equity": float("nan"),
            "total_return": float("nan"),
            "max_drawdown": float("nan"),
            "sharpe": float("nan"),
        }

    rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "name": name,
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "max_drawdown": _max_drawdown(equity),
        "sharpe": _annualized_sharpe_from_bar_returns(rets, bars_per_year=bars_per_year),
    }


def _infer_bars_per_year(results: pd.DataFrame) -> int:
    return 390 * 252


def _apply_switch_cost(returns: pd.Series, positions: pd.Series, switch_cost: float) -> pd.Series:
    position_changes = positions.diff().fillna(0.0).abs()
    switch_events = (position_changes > 0).astype(float)
    return returns - switch_events * switch_cost


def add_baselines(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()

    qqq_start = float(results["qqq_close"].iloc[0])
    tqqq_start = float(results["tqqq_close"].iloc[0])
    sqqq_start = float(results["sqqq_close"].iloc[0])

    results["qqq_equity"] = INITIAL_EQUITY * (results["qqq_close"] / max(qqq_start, 1e-8))
    results["tqqq_equity"] = INITIAL_EQUITY * (results["tqqq_close"] / max(tqqq_start, 1e-8))
    results["sqqq_equity"] = INITIAL_EQUITY * (results["sqqq_close"] / max(sqqq_start, 1e-8))

    results["tqqq_ret"] = results["tqqq_close"].pct_change().fillna(0.0)
    results["sqqq_ret"] = results["sqqq_close"].pct_change().fillna(0.0)

    # -------------------------------------------------
    # Regime baseline with one-bar lag to avoid lookahead
    # -------------------------------------------------
    regime_signal = results["regime"].shift(1).fillna("TRANSITION")

    regime_position = pd.Series(
        np.where(
            regime_signal.eq("BULL"),
            1.0,
            np.where(regime_signal.eq("BEAR"), -1.0, 0.0),
        ),
        index=results.index,
    )

    regime_ret = pd.Series(
        np.where(
            regime_position == 1.0,
            results["tqqq_ret"],
            np.where(regime_position == -1.0, results["sqqq_ret"], 0.0),
        ),
        index=results.index,
    )

    regime_ret = _apply_switch_cost(regime_ret, regime_position, BASELINE_SWITCH_COST)
    results["regime_switch_equity"] = INITIAL_EQUITY * (1.0 + regime_ret).cumprod()

    # -------------------------------------------------
    # SMA50 baseline with one-bar lag to avoid lookahead
    # above SMA50 => hold TQQQ
    # below SMA50 => hold SQQQ
    # -------------------------------------------------
    if "sma_50" in results.columns and results["sma_50"].notna().any():
        sma_signal = (results["qqq_close"].shift(1) > results["sma_50"].shift(1)).fillna(False)

        sma_position = pd.Series(
            np.where(sma_signal, 1.0, -1.0),
            index=results.index,
        )

        sma_ret = pd.Series(
            np.where(
                sma_position == 1.0,
                results["tqqq_ret"],
                results["sqqq_ret"],
            ),
            index=results.index,
        )

        sma_ret = _apply_switch_cost(sma_ret, sma_position, BASELINE_SWITCH_COST)
        results["sma50_switch_equity"] = INITIAL_EQUITY * (1.0 + sma_ret).cumprod()
    else:
        results["sma50_switch_equity"] = np.nan

    return results


def build_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    bars_per_year = _infer_bars_per_year(results)

    summaries = [
        _summarize_equity("strategy", results["strategy_equity"], bars_per_year),
        _summarize_equity("qqq_buy_hold", results["qqq_equity"], bars_per_year),
        _summarize_equity("tqqq_buy_hold", results["tqqq_equity"], bars_per_year),
        _summarize_equity("sqqq_buy_hold", results["sqqq_equity"], bars_per_year),
        _summarize_equity("regime_switch_baseline", results["regime_switch_equity"], bars_per_year),
    ]

    if "sma50_switch_equity" in results.columns:
        summaries.append(
            _summarize_equity("sma50_switch_baseline", results["sma50_switch_equity"], bars_per_year)
        )

    summary = pd.DataFrame(summaries)
    numeric_cols = ["final_equity", "total_return", "max_drawdown", "sharpe"]
    summary[numeric_cols] = summary[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return summary


# ---------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------
def run_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = download_last_3_months()

    trader = LiveTrader()
    feat = trader.build_features(raw)
    trader._load_vec_norm(feat)
    trader._reset_portfolio()

    print(f"[backtest] feature rows={len(feat):,}")
    print("[backtest] starting simulation...")

    records: list[dict[str, object]] = []

    for i, row in feat.iterrows():
        trader.global_step = i

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
                "action_label": str(action_label),
                "position_symbol": int(trader.position_symbol),
                "position_size": float(trader.position_size),
                "unrealized_pnl": float(trader.unrealized_pnl),
                "realized_pnl": float(trader.realized_pnl),
                "equity_norm": float(trader.equity),
                "regime": str(row["regime"]),
                "regime_conf": float(row["regime_conf"]),
                "signal_confidence": float(row["signal_confidence"]),
                "bull_score": float(row["bull_score"]),
                "bear_score": float(row["bear_score"]),
                "transition_score": float(row["transition_score"]),
                "sma_50": float(row["sma_50"]) if "sma_50" in row else np.nan,
            }
        )

    results = pd.DataFrame(records)
    if results.empty:
        raise RuntimeError("Backtest produced no records.")

    print("[backtest] action label distribution:")
    print(results["action_label"].value_counts(dropna=False).head(30).to_string())

    print("[backtest] regime distribution:")
    print(results["regime"].value_counts(dropna=False).to_string())

    print("[backtest] action id distribution:")
    print(results["action"].value_counts(dropna=False).sort_index().to_string())

    results["strategy_equity"] = INITIAL_EQUITY * results["equity_norm"]
    results = add_baselines(results)
    summary = build_summary_table(results)

    print("[backtest] finished.")
    print(summary.to_string(index=False, justify="left", float_format=lambda x: f"{x:,.4f}"))

    return results, summary


# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
def plot_results(results: pd.DataFrame) -> None:
    results = results.copy()
    results["timestamp"] = pd.to_datetime(results["timestamp"], errors="coerce")
    results = results.dropna(subset=["timestamp"]).reset_index(drop=True)

    numeric_cols = [
        "qqq_close",
        "tqqq_close",
        "sqqq_close",
        "strategy_equity",
        "qqq_equity",
        "tqqq_equity",
        "sqqq_equity",
        "regime_switch_equity",
        "sma50_switch_equity",
        "position_size",
    ]
    for col in numeric_cols:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")

    results = results.dropna(subset=["qqq_close", "strategy_equity", "qqq_equity"]).reset_index(drop=True)
    if results.empty:
        print("[backtest] no valid data to plot.")
        return

    strategy_start = float(results["strategy_equity"].iloc[0])
    results["strategy_equity_norm"] = results["strategy_equity"] / max(strategy_start, 1e-8)

    tqqq_buy_idx = results["action_label"].astype(str).str.contains("ENTER_TQQQ", na=False)
    tqqq_sell_idx = results["action_label"].astype(str).str.contains("EXIT_TQQQ", na=False)
    sqqq_buy_idx = results["action_label"].astype(str).str.contains("ENTER_SQQQ", na=False)
    sqqq_sell_idx = results["action_label"].astype(str).str.contains("EXIT_SQQQ", na=False)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    ax_price = axes[0]
    ax_price.plot(results["timestamp"], results["qqq_close"], linewidth=1.5, label="QQQ Price")
    ax_price.set_title("QQQ Price Action with Strategy Equity Overlay")
    ax_price.grid(alpha=0.3)

    ax_eq = ax_price.twinx()
    ax_eq.plot(
        results["timestamp"],
        results["strategy_equity_norm"],
        linewidth=1.5,
        alpha=0.85,
        label="Strategy Equity (normalized)",
    )

    lines1, labels1 = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_eq.get_legend_handles_labels()
    ax_price.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    axes[1].plot(results["timestamp"], results["strategy_equity"], linewidth=2, label="Strategy")
    axes[1].plot(results["timestamp"], results["qqq_equity"], linewidth=1.5, label="QQQ Buy & Hold")
    axes[1].plot(results["timestamp"], results["regime_switch_equity"], linewidth=1.5, label="Regime Switch Baseline")

    if "sma50_switch_equity" in results.columns and results["sma50_switch_equity"].notna().any():
        axes[1].plot(
            results["timestamp"],
            results["sma50_switch_equity"],
            linewidth=1.5,
            label="SMA50 Switch Baseline",
        )

    axes[1].set_title("Strategy Equity vs Baselines")
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].grid(alpha=0.3)

    axes[2].plot(results["timestamp"], results["tqqq_equity"], linewidth=1.5, label="TQQQ Buy & Hold")
    axes[2].plot(results["timestamp"], results["sqqq_equity"], linewidth=1.5, label="SQQQ Buy & Hold")
    axes[2].plot(results["timestamp"], results["qqq_equity"], linewidth=1.5, label="QQQ Buy & Hold")
    axes[2].set_title("Instrument Buy-and-Hold Benchmarks")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    axes[3].plot(results["timestamp"], results["qqq_close"], linewidth=1.0, alpha=0.55, label="QQQ Price")
    axes[3].scatter(results.loc[tqqq_buy_idx, "timestamp"], results.loc[tqqq_buy_idx, "qqq_close"], s=20, label="TQQQ Entry")
    axes[3].scatter(results.loc[tqqq_sell_idx, "timestamp"], results.loc[tqqq_sell_idx, "qqq_close"], s=20, label="TQQQ Exit")
    axes[3].scatter(results.loc[sqqq_buy_idx, "timestamp"], results.loc[sqqq_buy_idx, "qqq_close"], s=20, label="SQQQ Entry")
    axes[3].scatter(results.loc[sqqq_sell_idx, "timestamp"], results.loc[sqqq_sell_idx, "qqq_close"], s=20, label="SQQQ Exit")
    axes[3].set_title("Trade Actions on QQQ Price")
    axes[3].legend(loc="upper left", ncol=2)
    axes[3].grid(alpha=0.3)

    if "position_size" in results.columns and results["position_size"].notna().any():
        ax_size = axes[3].twinx()
        ax_size.plot(
            results["timestamp"],
            results["position_size"],
            linewidth=1.2,
            alpha=0.7,
            label="Position Size",
        )
        ax_size.set_ylim(-0.05, 1.05)

        lines3, labels3 = axes[3].get_legend_handles_labels()
        lines4, labels4 = ax_size.get_legend_handles_labels()
        axes[3].legend(lines3 + lines4, labels3 + labels4, loc="upper left", ncol=3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_df, summary_df = run_backtest()
    plot_results(results_df)