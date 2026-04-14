from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from momentum_predictor.pipeline import load_data


EQUITY_PATH = "outputs/backtests/tqqq_sqqq_filter_equity.csv"
TRADES_PATH = "outputs/backtests/tqqq_sqqq_filter_trades.csv"
INITIAL_CAPITAL = 10_000.0
TZ = "America/New_York"


def normalize_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(TZ)


def build_qqq_baseline(start_ts: pd.Timestamp, end_ts: pd.Timestamp, initial_capital: float) -> pd.DataFrame:
    qqq = load_data("QQQ", start_ts.isoformat(), end_ts.isoformat()).copy()
    qqq["timestamp"] = normalize_ts(qqq["timestamp"])
    qqq = qqq.sort_values("timestamp").reset_index(drop=True)

    if qqq.empty:
        raise ValueError("QQQ baseline data is empty.")

    start_price = float(qqq.iloc[0]["close"])
    qqq["qqq_equity"] = initial_capital * (qqq["close"].astype(float) / start_price)
    return qqq[["timestamp", "qqq_equity"]]


def main():
    equity = pd.read_csv(EQUITY_PATH)
    trades = pd.read_csv(TRADES_PATH)

    equity["timestamp"] = normalize_ts(equity["timestamp"])
    equity = equity.sort_values("timestamp").reset_index(drop=True)
    equity["equity"] = equity["equity"].astype(float)

    if not trades.empty:
        trades["entry_time"] = normalize_ts(trades["entry_time"])
        trades["exit_time"] = normalize_ts(trades["exit_time"])
        trades["entry_price"] = trades["entry_price"].astype(float)
        trades["exit_price"] = trades["exit_price"].astype(float)

    start_ts = equity["timestamp"].min()
    end_ts = equity["timestamp"].max()

    qqq = build_qqq_baseline(start_ts, end_ts, INITIAL_CAPITAL)

    merged = pd.merge_asof(
        equity[["timestamp", "equity"]].sort_values("timestamp"),
        qqq.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    # ----------------------------------------
    # EQUITY + BASELINE
    # ----------------------------------------
    ax.plot(merged["timestamp"], merged["equity"], label="Strategy Equity")
    ax.plot(merged["timestamp"], merged["qqq_equity"], linestyle="--", label="QQQ Baseline")

    # ----------------------------------------
    # TRADE OVERLAY (TQQQ + SQQQ)
    # ----------------------------------------
    if not trades.empty:
        entry_points = pd.merge_asof(
            trades[["entry_time", "symbol"]]
            .rename(columns={"entry_time": "timestamp"})
            .sort_values("timestamp"),
            merged.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        exit_points = pd.merge_asof(
            trades[["exit_time", "symbol"]]
            .rename(columns={"exit_time": "timestamp"})
            .sort_values("timestamp"),
            merged.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        # Split by symbol
        tqqq_entries = entry_points[entry_points["symbol"] == "TQQQ"]
        sqqq_entries = entry_points[entry_points["symbol"] == "SQQQ"]

        tqqq_exits = exit_points[exit_points["symbol"] == "TQQQ"]
        sqqq_exits = exit_points[exit_points["symbol"] == "SQQQ"]

        # TQQQ (long)
        if not tqqq_entries.empty:
            ax.scatter(
                tqqq_entries["timestamp"],
                tqqq_entries["equity"],
                marker="^",
                s=80,
                label="TQQQ Entry",
            )

        if not tqqq_exits.empty:
            ax.scatter(
                tqqq_exits["timestamp"],
                tqqq_exits["equity"],
                marker="x",
                s=70,
                label="TQQQ Exit",
            )

        # SQQQ (inverse / short)
        if not sqqq_entries.empty:
            ax.scatter(
                sqqq_entries["timestamp"],
                sqqq_entries["equity"],
                marker="v",
                s=80,
                label="SQQQ Entry",
            )

        if not sqqq_exits.empty:
            ax.scatter(
                sqqq_exits["timestamp"],
                sqqq_exits["equity"],
                marker="x",
                s=70,
                label="SQQQ Exit",
            )

    ax.set_title("Strategy (TQQQ + SQQQ) vs QQQ Baseline")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity ($)")
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()