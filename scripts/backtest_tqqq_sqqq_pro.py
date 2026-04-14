from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from momentum_predictor.pipeline import load_data


PREDICTIONS_PATH = Path("outputs/signals/predictions.csv")
TRADES_PATH = Path("outputs/backtests/tqqq_sqqq_adaptive_trades.csv")
EQUITY_PATH = Path("outputs/backtests/tqqq_sqqq_adaptive_equity.csv")

INITIAL_CASH = 10_000.0
LONG_CONF_THRESHOLD = 0.449
SHORT_CONF_THRESHOLD = 0.659
COMMISSION_PER_TRADE = 0.0


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    return_pct: float
    exit_reason: str


# =========================================================
# DATA HELPERS
# =========================================================
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


# =========================================================
# FILTER FEATURES
# =========================================================
def compute_filter_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    df["sma20"] = df["close"].rolling(20).mean()
    df["sma20_d5"] = df["sma20"].shift(5)
    df["sma50"] = df["close"].rolling(50).mean()

    return df


def get_trade_mode(row: pd.Series) -> str:
    close = row.get("close")
    sma20 = row.get("sma20")
    sma20_d5 = row.get("sma20_d5")
    sma50 = row.get("sma50")

    if pd.isna(close) or pd.isna(sma20) or pd.isna(sma20_d5) or pd.isna(sma50):
        return "flat"

    # Bearish condition gets priority
    if sma20_d5 < sma20:
        return "short_only"

    # Bullish condition
    if close > sma50:
        return "long_only"

    return "flat"


def _latest_filter_row(filter_bars: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    idx = filter_bars["timestamp"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return None
    return filter_bars.iloc[idx]


# =========================================================
# BACKTEST
# =========================================================
def run_backtest(
    predictions_path: str | Path = PREDICTIONS_PATH,
    initial_cash: float = INITIAL_CASH,
    long_conf_threshold: float = LONG_CONF_THRESHOLD,
    short_conf_threshold: float = SHORT_CONF_THRESHOLD,
    commission_per_trade: float = COMMISSION_PER_TRADE,
):
    preds = pd.read_csv(predictions_path).copy()
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True).dt.tz_convert("America/New_York")
    preds = preds.sort_values("timestamp").reset_index(drop=True)

    if preds.empty:
        raise ValueError("predictions.csv is empty.")

    start = preds["timestamp"].min().floor("D")
    end = preds["timestamp"].max().ceil("D") + pd.Timedelta(days=1)

    tqqq_bars = _prepare_bars("TQQQ", start, end)
    sqqq_bars = _prepare_bars("SQQQ", start, end)
    qqq_filter_bars = _prepare_bars("QQQ", start, end)
    qqq_filter_bars = compute_filter_features(qqq_filter_bars)

    cash = float(initial_cash)
    shares = 0.0
    position_symbol: str | None = None
    entry_price = None
    entry_time = None

    trades: list[Trade] = []
    equity_rows: list[dict] = []

    blocked_long_entries = 0
    blocked_short_entries = 0
    taken_long_entries = 0
    taken_short_entries = 0
    filter_liquidations = 0
    signal_liquidations = 0

    for _, row in preds.iterrows():
        ts = row["timestamp"]
        signal = str(row["signal"]).upper()
        confidence = float(row["confidence"])

        long_trigger = signal == "LONG" and confidence > long_conf_threshold
        short_trigger = signal == "SHORT" and confidence > short_conf_threshold

        filter_row = _latest_filter_row(qqq_filter_bars, ts)
        trade_mode = "flat" if filter_row is None else get_trade_mode(filter_row)

        long_allowed = trade_mode == "long_only"
        short_allowed = trade_mode == "short_only"

        # -------------------------------------------------
        # FILTER-BASED LIQUIDATIONS
        # -------------------------------------------------
        if position_symbol == "TQQQ" and not long_allowed:
            exec_ts, exec_px = _next_bar_open(tqqq_bars, ts)
            if exec_px is not None:
                proceeds = shares * exec_px
                cash += proceeds - commission_per_trade

                pnl = (exec_px - entry_price) * shares - 2 * commission_per_trade
                ret_pct = (exec_px / entry_price) - 1.0

                trades.append(
                    Trade(
                        entry_time=str(entry_time),
                        exit_time=str(exec_ts),
                        symbol="TQQQ",
                        side="LONG",
                        entry_price=float(entry_price),
                        exit_price=float(exec_px),
                        shares=float(shares),
                        pnl=float(pnl),
                        return_pct=float(ret_pct),
                        exit_reason="filter_no_long_allowed",
                    )
                )

                shares = 0.0
                position_symbol = None
                entry_price = None
                entry_time = None
                filter_liquidations += 1

        elif position_symbol == "SQQQ" and not short_allowed:
            exec_ts, exec_px = _next_bar_open(sqqq_bars, ts)
            if exec_px is not None:
                proceeds = shares * exec_px
                cash += proceeds - commission_per_trade

                pnl = (exec_px - entry_price) * shares - 2 * commission_per_trade
                ret_pct = (exec_px / entry_price) - 1.0

                trades.append(
                    Trade(
                        entry_time=str(entry_time),
                        exit_time=str(exec_ts),
                        symbol="SQQQ",
                        side="LONG_INVERSE",
                        entry_price=float(entry_price),
                        exit_price=float(exec_px),
                        shares=float(shares),
                        pnl=float(pnl),
                        return_pct=float(ret_pct),
                        exit_reason="filter_no_short_allowed",
                    )
                )

                shares = 0.0
                position_symbol = None
                entry_price = None
                entry_time = None
                filter_liquidations += 1

        # -------------------------------------------------
        # SIGNAL-BASED EXITS
        # -------------------------------------------------
        if position_symbol == "TQQQ" and short_trigger:
            exec_ts, exec_px = _next_bar_open(tqqq_bars, ts)
            if exec_px is not None:
                proceeds = shares * exec_px
                cash += proceeds - commission_per_trade

                pnl = (exec_px - entry_price) * shares - 2 * commission_per_trade
                ret_pct = (exec_px / entry_price) - 1.0

                trades.append(
                    Trade(
                        entry_time=str(entry_time),
                        exit_time=str(exec_ts),
                        symbol="TQQQ",
                        side="LONG",
                        entry_price=float(entry_price),
                        exit_price=float(exec_px),
                        shares=float(shares),
                        pnl=float(pnl),
                        return_pct=float(ret_pct),
                        exit_reason="model_short_signal",
                    )
                )

                shares = 0.0
                position_symbol = None
                entry_price = None
                entry_time = None
                signal_liquidations += 1

        elif position_symbol == "SQQQ" and long_trigger:
            exec_ts, exec_px = _next_bar_open(sqqq_bars, ts)
            if exec_px is not None:
                proceeds = shares * exec_px
                cash += proceeds - commission_per_trade

                pnl = (exec_px - entry_price) * shares - 2 * commission_per_trade
                ret_pct = (exec_px / entry_price) - 1.0

                trades.append(
                    Trade(
                        entry_time=str(entry_time),
                        exit_time=str(exec_ts),
                        symbol="SQQQ",
                        side="LONG_INVERSE",
                        entry_price=float(entry_price),
                        exit_price=float(exec_px),
                        shares=float(shares),
                        pnl=float(pnl),
                        return_pct=float(ret_pct),
                        exit_reason="model_long_signal",
                    )
                )

                shares = 0.0
                position_symbol = None
                entry_price = None
                entry_time = None
                signal_liquidations += 1

        # -------------------------------------------------
        # NEW ENTRIES
        # -------------------------------------------------
        if position_symbol is None:
            if long_trigger:
                if long_allowed:
                    exec_ts, exec_px = _next_bar_open(tqqq_bars, ts)
                    if exec_px is not None and cash > commission_per_trade:
                        deployable_cash = cash - commission_per_trade
                        shares = deployable_cash / exec_px
                        cash -= shares * exec_px + commission_per_trade

                        position_symbol = "TQQQ"
                        entry_price = exec_px
                        entry_time = exec_ts
                        taken_long_entries += 1
                else:
                    blocked_long_entries += 1

            elif short_trigger:
                if short_allowed:
                    exec_ts, exec_px = _next_bar_open(sqqq_bars, ts)
                    if exec_px is not None and cash > commission_per_trade:
                        deployable_cash = cash - commission_per_trade
                        shares = deployable_cash / exec_px
                        cash -= shares * exec_px + commission_per_trade

                        position_symbol = "SQQQ"
                        entry_price = exec_px
                        entry_time = exec_ts
                        taken_short_entries += 1
                else:
                    blocked_short_entries += 1

        # -------------------------------------------------
        # EQUITY TRACKING
        # -------------------------------------------------
        if position_symbol == "TQQQ":
            px = _latest_close(tqqq_bars, ts)
            equity = cash + (shares * px if px is not None else 0.0)
        elif position_symbol == "SQQQ":
            px = _latest_close(sqqq_bars, ts)
            equity = cash + (shares * px if px is not None else 0.0)
        else:
            equity = cash

        equity_rows.append(
            {
                "timestamp": ts,
                "cash": cash,
                "shares": shares,
                "equity": equity,
                "signal": signal,
                "confidence": confidence,
                "position_symbol": position_symbol if position_symbol else "FLAT",
                "trade_mode": trade_mode,
                "close": None if filter_row is None else filter_row["close"],
                "sma20": None if filter_row is None else filter_row["sma20"],
                "sma20_d5": None if filter_row is None else filter_row["sma20_d5"],
                "sma50": None if filter_row is None else filter_row["sma50"],
            }
        )

    # -----------------------------------------------------
    # FINAL LIQUIDATION
    # -----------------------------------------------------
    final_ts = preds["timestamp"].max()

    if position_symbol == "TQQQ" and shares > 0:
        final_px = _latest_close(tqqq_bars, final_ts)
        if final_px is not None:
            cash += shares * final_px - commission_per_trade
            pnl = (final_px - entry_price) * shares - 2 * commission_per_trade
            ret_pct = (final_px / entry_price) - 1.0
            trades.append(
                Trade(
                    entry_time=str(entry_time),
                    exit_time=str(final_ts),
                    symbol="TQQQ",
                    side="LONG",
                    entry_price=float(entry_price),
                    exit_price=float(final_px),
                    shares=float(shares),
                    pnl=float(pnl),
                    return_pct=float(ret_pct),
                    exit_reason="final_liquidation",
                )
            )
            shares = 0.0
            position_symbol = None

    elif position_symbol == "SQQQ" and shares > 0:
        final_px = _latest_close(sqqq_bars, final_ts)
        if final_px is not None:
            cash += shares * final_px - commission_per_trade
            pnl = (final_px - entry_price) * shares - 2 * commission_per_trade
            ret_pct = (final_px / entry_price) - 1.0
            trades.append(
                Trade(
                    entry_time=str(entry_time),
                    exit_time=str(final_ts),
                    symbol="SQQQ",
                    side="LONG_INVERSE",
                    entry_price=float(entry_price),
                    exit_price=float(final_px),
                    shares=float(shares),
                    pnl=float(pnl),
                    return_pct=float(ret_pct),
                    exit_reason="final_liquidation",
                )
            )
            shares = 0.0
            position_symbol = None

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    equity_df = pd.DataFrame(equity_rows)

    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(TRADES_PATH, index=False)
    equity_df.to_csv(EQUITY_PATH, index=False)

    final_equity = cash
    total_return = (final_equity / initial_cash) - 1.0

    if not equity_df.empty:
        eq = equity_df["equity"].astype(float)
        running_max = eq.cummax()
        drawdown = (eq / running_max) - 1.0
        max_drawdown = float(drawdown.min())
    else:
        max_drawdown = 0.0

    if not trades_df.empty:
        win_rate = float((trades_df["pnl"] > 0).mean())
        avg_trade = float(trades_df["pnl"].mean())
        n_trades = int(len(trades_df))
    else:
        win_rate = 0.0
        avg_trade = 0.0
        n_trades = 0

    print("\n=== BACKTEST SUMMARY ===")
    print(f"initial_cash:          {initial_cash:,.2f}")
    print(f"final_equity:          {final_equity:,.2f}")
    print(f"total_return:          {total_return:.2%}")
    print(f"max_drawdown:          {max_drawdown:.2%}")
    print(f"num_trades:            {n_trades}")
    print(f"win_rate:              {win_rate:.2%}")
    print(f"avg_trade_pnl:         {avg_trade:,.2f}")
    print(f"blocked_long_entries:  {blocked_long_entries}")
    print(f"blocked_short_entries: {blocked_short_entries}")
    print(f"taken_long_entries:    {taken_long_entries}")
    print(f"taken_short_entries:   {taken_short_entries}")
    print(f"filter_liquidations:   {filter_liquidations}")
    print(f"signal_liquidations:   {signal_liquidations}")
    print(f"trades_csv:            {TRADES_PATH}")
    print(f"equity_csv:            {EQUITY_PATH}")

    return trades_df, equity_df


if __name__ == "__main__":
    run_backtest()