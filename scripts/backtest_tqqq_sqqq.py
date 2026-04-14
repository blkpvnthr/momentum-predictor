from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from momentum_predictor.pipeline import load_data


PREDICTIONS_PATH = Path("outputs/signals/predictions.csv")
TRADES_PATH = Path("outputs/backtests/sqqq_trades.csv")
EQUITY_PATH = Path("outputs/backtests/sqqq_equity.csv")

INITIAL_CASH = 10_000.0
SHORT_CONF_THRESHOLD = 0.69
COMMISSION_PER_TRADE = 0.0


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    symbol: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    return_pct: float


# ----------------------------------------
# DATA HELPERS
# ----------------------------------------
def _prepare_bars(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = load_data(symbol, start.isoformat(), end.isoformat()).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    return df.sort_values("timestamp").reset_index(drop=True)


def _next_bar_open(bars: pd.DataFrame, ts: pd.Timestamp):
    idx = bars["timestamp"].searchsorted(ts, side="right")
    if idx >= len(bars):
        return None, None
    row = bars.iloc[idx]
    return row["timestamp"], float(row["open"])


def _latest_close(bars: pd.DataFrame, ts: pd.Timestamp):
    idx = bars["timestamp"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return None
    return float(bars.iloc[idx]["close"])


# ----------------------------------------
# BACKTEST
# ----------------------------------------
def run_backtest():
    preds = pd.read_csv(PREDICTIONS_PATH)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True).dt.tz_convert("America/New_York")
    preds = preds.sort_values("timestamp").reset_index(drop=True)

    if preds.empty:
        raise ValueError("predictions.csv is empty.")

    start = preds["timestamp"].min().floor("D")
    end = preds["timestamp"].max().ceil("D") + pd.Timedelta(days=1)

    sqqq_bars = _prepare_bars("SQQQ", start, end)

    cash = INITIAL_CASH
    shares = 0.0
    in_position = False
    entry_price = None
    entry_time = None

    trades = []
    equity_rows = []

    for _, row in preds.iterrows():
        ts = row["timestamp"]
        signal = str(row["signal"]).upper()
        confidence = float(row["confidence"])

        short_trigger = signal == "SHORT" and confidence > SHORT_CONF_THRESHOLD
        exit_trigger = signal == "LONG" and confidence > 0.449

        # ENTRY (short via SQQQ)
        if not in_position and short_trigger:
            exec_ts, exec_px = _next_bar_open(sqqq_bars, ts)
            if exec_px is not None:
                shares = cash / exec_px
                cash -= shares * exec_px
                in_position = True
                entry_price = exec_px
                entry_time = exec_ts

        # EXIT
        elif in_position and exit_trigger:
            exec_ts, exec_px = _next_bar_open(sqqq_bars, ts)
            if exec_px is not None:
                proceeds = shares * exec_px
                cash += proceeds

                pnl = (exec_px - entry_price) * shares
                ret = exec_px / entry_price - 1

                trades.append(
                    Trade(
                        entry_time=str(entry_time),
                        exit_time=str(exec_ts),
                        symbol="SQQQ",
                        entry_price=entry_price,
                        exit_price=exec_px,
                        shares=shares,
                        pnl=pnl,
                        return_pct=ret,
                    )
                )

                shares = 0.0
                in_position = False

        # EQUITY TRACKING
        if in_position:
            px = _latest_close(sqqq_bars, ts)
            equity = cash + (shares * px if px else 0)
        else:
            equity = cash

        equity_rows.append(
            {
                "timestamp": ts,
                "equity": equity,
                "signal": signal,
                "confidence": confidence,
            }
        )

    # FINAL CLOSE
    if in_position:
        final_px = _latest_close(sqqq_bars, preds["timestamp"].iloc[-1])
        if final_px:
            cash += shares * final_px

    # SAVE
    trades_df = pd.DataFrame([asdict(t) for t in trades])
    equity_df = pd.DataFrame(equity_rows)

    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(TRADES_PATH, index=False)
    equity_df.to_csv(EQUITY_PATH, index=False)

    print("\n=== BACKTEST SUMMARY ===")
    print(f"final_equity: {cash:,.2f}")
    print(f"trades: {len(trades_df)}")

    return trades_df, equity_df


# ----------------------------------------
# DRAWdown plot
# ----------------------------------------
def plot_drawdown():
    df = pd.read_csv(EQUITY_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")

    eq = df["equity"].astype(float)
    dd = eq / eq.cummax() - 1

    plt.figure()
    plt.plot(df["timestamp"], dd)
    plt.title("Drawdown (SQQQ only)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_backtest()
    plot_drawdown()