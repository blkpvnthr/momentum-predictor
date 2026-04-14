from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderStatus, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from trading_system import LiveTrader


THIS_FILE = Path(__file__).resolve()
LIVE_DIR = THIS_FILE.parent
PROJECT_ROOT = LIVE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

LOG_DIR = LIVE_DIR / "logs"
JOURNAL_PATH = LOG_DIR / "paper_trader_journal.csv"
EQUITY_CURVE_PATH = LOG_DIR / "paper_trader_equity_curve.csv"

TIMEZONE = "America/New_York"

SIGNAL_SYMBOL = "QQQ"
FEATURE_SYMBOLS = ("QQQ", "TQQQ", "SQQQ")

BULL_UNIVERSE = (
    "TQQQ",
    "SATL",
    "LWLG",
    "TE",
    "DVLT",
    "ALOY",
    "NIKA",
    "SPIR",
    "MRLN",
    "STAA",
    "LPTH",
    "ENVX",
    "CTGO",
    "OPTX",
)

BEAR_UNIVERSE = (
    "SDOW",
    "SQQQ",
    "SRTY",
    "REW",
    "SOXS",
    "SPXU",
    "DOG",
    "DXD",
    "NVD",
)

EXEC_SYMBOLS = tuple(sorted(set(BULL_UNIVERSE) | set(BEAR_UNIVERSE)))
ALL_SYMBOLS = (SIGNAL_SYMBOL,) + EXEC_SYMBOLS

BAR_TIMEFRAME = TimeFrame.Minute
WARMUP_BARS = 500

TARGET_DOLLARS = 5000.0
MIN_BUYING_POWER = 250.0
ORDER_WAIT_SECONDS = 20
COOLDOWN_SECONDS = 20

MIN_SIGNAL_CONFIDENCE = 0.45
MIN_REGIME_CONFIDENCE = 0.50
MIN_REENTRY_SECONDS = 15
ALLOW_TRANSITION_ENTRIES = False
TRANSITION_SIZE_MULTIPLIER = 0.35
HEARTBEAT_SECONDS = 60

TOP_CANDIDATES_TO_PRINT = 5
MIN_CANDIDATE_PRICE = 1.00
MIN_MARK_PCT_CHG_BULL = 0.001
MIN_MARK_PCT_CHG_BEAR = 0.001

FALLBACK_BULL_MARK_PCT_CHG = 0.0030
FALLBACK_BEAR_MARK_PCT_CHG = 0.0030

MAX_LOSS_PNL_PCT = -0.01

MAX_BULL_POSITIONS = 5
MAX_BEAR_POSITIONS = 5


@dataclass
class BrokerPosition:
    symbol: Optional[str] = None
    qty: float = 0.0
    market_value: float = 0.0
    avg_entry_price: float = 0.0


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class CsvJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = [
            "timestamp_utc",
            "timestamp_et",
            "bar_timestamp_et",
            "regime",
            "regime_conf",
            "signal_confidence",
            "bull_score",
            "bear_score",
            "action_label",
            "selected_symbol",
            "selection_side",
            "selection_rank_score",
            "selection_mark_pct_chg",
            "bull_positions_before",
            "bear_positions_before",
            "bull_positions_after",
            "bear_positions_after",
            "qqq_close",
            "buying_power_before",
            "equity_before",
            "cash_before",
            "buying_power_after",
            "equity_after",
            "cash_after",
            "trade_side",
            "trade_symbol",
            "trade_qty",
            "trade_price",
            "confidence_alloc_frac",
            "kelly_frac",
            "final_alloc_dollars",
            "entry_price",
            "exit_price",
            "realized_pnl",
            "realized_pnl_pct",
            "message",
        ]
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.columns).writeheader()

    def append(self, row: dict[str, object]) -> None:
        safe_row = {k: row.get(k, "") for k in self.columns}
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.columns).writerow(safe_row)


class AlpacaPaperBroker:
    def __init__(self, env_path: Path):
        load_dotenv(env_path)

        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in .env")

        self.market_data = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )
        self.trading = TradingClient(self.api_key, self.secret_key, paper=True)

    def create_stream(self) -> StockDataStream:
        return StockDataStream(self.api_key, self.secret_key, feed=DataFeed.IEX)

    def get_account_snapshot(self) -> dict[str, float]:
        acct = self.trading.get_account()
        return {
            "buying_power": float(acct.buying_power),
            "equity": float(acct.equity),
            "cash": float(acct.cash),
        }

    def get_position(self, symbol: str) -> BrokerPosition:
        try:
            pos = self.trading.get_open_position(symbol)
            return BrokerPosition(
                symbol=str(symbol),
                qty=float(pos.qty),
                market_value=float(pos.market_value),
                avg_entry_price=float(pos.avg_entry_price),
            )
        except Exception:
            return BrokerPosition()

    def get_all_open_positions(self) -> list[BrokerPosition]:
        positions: list[BrokerPosition] = []
        try:
            raw_positions = self.trading.get_all_positions()
        except Exception:
            return positions

        for pos in raw_positions:
            try:
                qty = float(pos.qty)
            except Exception:
                qty = 0.0

            if qty <= 0:
                continue

            try:
                positions.append(
                    BrokerPosition(
                        symbol=str(pos.symbol),
                        qty=qty,
                        market_value=float(pos.market_value),
                        avg_entry_price=float(pos.avg_entry_price),
                    )
                )
            except Exception:
                continue

        return positions

    def get_positions_for_symbols(self, symbols: tuple[str, ...] | list[str]) -> list[BrokerPosition]:
        symbol_set = set(symbols)
        return [p for p in self.get_all_open_positions() if p.symbol in symbol_set]

    def has_position(self, symbol: str) -> bool:
        return self.get_position(symbol).qty > 0

    def list_open_orders(self) -> list:
        try:
            return self.trading.get_orders()
        except Exception:
            return []

    def has_open_order_for_symbols(self, symbols: tuple[str, ...] | list[str]) -> bool:
        open_statuses = {
            OrderStatus.NEW,
            OrderStatus.ACCEPTED,
            OrderStatus.PENDING_NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.ACCEPTED_FOR_BIDDING,
            OrderStatus.CALCULATED,
        }
        symbols_set = set(symbols)
        for order in self.list_open_orders():
            try:
                if order.symbol in symbols_set and order.status in open_statuses:
                    return True
            except Exception:
                continue
        return False

    def submit_market_buy(self, symbol: str, qty: int):
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side="buy",
            time_in_force=TimeInForce.DAY,
        )
        return self.trading.submit_order(order_data=order)

    def submit_market_sell(self, symbol: str, qty: int):
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side="sell",
            time_in_force=TimeInForce.DAY,
        )
        return self.trading.submit_order(order_data=order)

    def wait_until_flat(self, symbol: str, timeout_seconds: int = ORDER_WAIT_SECONDS) -> None:
        started = time.time()
        while time.time() - started < timeout_seconds:
            if self.get_position(symbol).qty <= 0:
                return
            time.sleep(0.5)
        raise TimeoutError(f"{symbol} position did not flatten within timeout")

    def wait_for_position(self, symbol: str, timeout_seconds: int = 10) -> BrokerPosition:
        started = time.time()
        while time.time() - started < timeout_seconds:
            pos = self.get_position(symbol)
            if pos.qty > 0:
                return pos
            time.sleep(0.5)
        return BrokerPosition()

    def close_symbol_if_open(self, symbol: str) -> None:
        pos = self.get_position(symbol)
        if pos.qty > 0:
            self.submit_market_sell(symbol, int(pos.qty))
            self.wait_until_flat(symbol)

    def fetch_recent_symbol_bars(
        self,
        symbols: list[str],
        lookback_bars: int = WARMUP_BARS,
    ) -> pd.DataFrame:
        end_ts = now_utc()
        start_ts = end_ts - timedelta(days=10)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=BAR_TIMEFRAME,
            start=start_ts,
            end=end_ts,
            feed=DataFeed.IEX,
        )
        bars = self.market_data.get_stock_bars(request).df
        if bars is None or len(bars) == 0:
            raise RuntimeError("No bar data returned from Alpaca")

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
            raise RuntimeError(f"Bars missing columns: {missing}")

        bars["timestamp"] = (
            pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
            .dt.tz_convert(TIMEZONE)
            .dt.floor("min")
        )
        bars = bars.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        out_frames: list[pd.DataFrame] = []
        for symbol in symbols:
            sdf = bars[bars["symbol"] == symbol].copy()
            if len(sdf) == 0:
                continue
            out_frames.append(sdf.tail(lookback_bars))

        if not out_frames:
            raise RuntimeError("No per-symbol bars returned after filtering")

        return pd.concat(out_frames, ignore_index=True)


class ProductionPaperTrader:
    def __init__(
        self,
        target_dollars: float = TARGET_DOLLARS,
        cooldown_seconds: int = COOLDOWN_SECONDS,
    ):
        self.target_dollars = float(target_dollars)
        self.cooldown_seconds = int(cooldown_seconds)

        self.journal = CsvJournal(JOURNAL_PATH)
        self.broker = AlpacaPaperBroker(ENV_PATH)
        self.strategy = LiveTrader()

        self.last_order_time: float = 0.0
        self.last_entry_time: float = 0.0
        self.last_heartbeat_time: float = 0.0

        self.open_trades: dict[str, dict[str, float | str]] = {}
        self.closed_trades: list[dict[str, float | str]] = []
        self.default_win_rate = 0.55
        self.default_win_loss_ratio = 1.50

        self.account_history: list[dict[str, float | str]] = []
        self.last_equity: float | None = None

        self.live_symbol_frames: dict[str, pd.DataFrame] = {
            symbol: pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            for symbol in ALL_SYMBOLS
        }
        self.last_processed_timestamp: Optional[pd.Timestamp] = None

    def _in_cooldown(self) -> bool:
        return (time.time() - self.last_order_time) < self.cooldown_seconds

    def _in_reentry_lockout(self) -> bool:
        return (time.time() - self.last_entry_time) < MIN_REENTRY_SECONDS

    def _mark_order(self) -> None:
        self.last_order_time = time.time()

    def _mark_entry_time(self) -> None:
        self.last_entry_time = time.time()

    def _is_regular_hours_bar(self, ts: pd.Timestamp) -> bool:
        if ts.tzinfo is None:
            return False
        hhmm = ts.hour * 100 + ts.minute
        return 930 <= hhmm <= 1600

    def _desired_action_from_signal(self, action_label: str) -> str:
        allowed = {
            "ENTER_TQQQ",
            "ENTER_TQQQ_TRANSITION",
            "EXIT_TQQQ",
            "FORCED_EXIT_TQQQ",
            "ENTER_SQQQ",
            "ENTER_SQQQ_TRANSITION",
            "EXIT_SQQQ",
            "FORCED_EXIT_SQQQ",
            "HOLD",
        }
        return action_label if action_label in allowed else "HOLD"

    def _record_account_snapshot(
        self,
        label: str,
        bar_ts: pd.Timestamp | None = None,
    ) -> dict[str, float | str]:
        acct = self.broker.get_account_snapshot()
        snapshot = {
            "label": label,
            "timestamp_utc": now_utc().isoformat(),
            "bar_timestamp_et": "" if bar_ts is None or pd.isna(bar_ts) else bar_ts.isoformat(),
            "buying_power": float(acct["buying_power"]),
            "equity": float(acct["equity"]),
            "cash": float(acct["cash"]),
        }
        self.account_history.append(snapshot)
        self.last_equity = float(acct["equity"])
        return snapshot

    def _write_account_equity_curve(self) -> None:
        if self.account_history:
            pd.DataFrame(self.account_history).to_csv(EQUITY_CURVE_PATH, index=False)

    def _latest_symbol_price(self, symbol: str, ts: pd.Timestamp) -> float | None:
        df = self.live_symbol_frames.get(symbol)
        if df is None or len(df) == 0:
            return None

        latest = df[df["timestamp"] <= ts].tail(1)
        if len(latest) == 0:
            return None

        price = latest["close"].iloc[-1]
        if pd.isna(price):
            return None
        return float(price)

    def _mark_pct_change(self, symbol: str, ts: pd.Timestamp) -> float | None:
        df = self.live_symbol_frames.get(symbol)
        if df is None or len(df) < 2:
            return None

        sdf = df[df["timestamp"] <= ts].sort_values("timestamp").tail(2)
        if len(sdf) < 2:
            return None

        prev_close = sdf["close"].iloc[-2]
        last_close = sdf["close"].iloc[-1]

        if pd.isna(prev_close) or pd.isna(last_close) or float(prev_close) <= 0:
            return None

        return (float(last_close) - float(prev_close)) / float(prev_close)

    def _kelly_fraction(self) -> float:
        if len(self.closed_trades) < 10:
            p = self.default_win_rate
            b = self.default_win_loss_ratio
        else:
            wins = [t for t in self.closed_trades if float(t.get("realized_pnl", 0.0)) > 0]
            losses = [t for t in self.closed_trades if float(t.get("realized_pnl", 0.0)) <= 0]

            total = len(self.closed_trades)
            p = len(wins) / total if total > 0 else self.default_win_rate

            avg_win = (
                sum(float(t["realized_pnl"]) for t in wins) / len(wins)
                if wins else 0.0
            )
            avg_loss = (
                abs(sum(float(t["realized_pnl"]) for t in losses) / len(losses))
                if losses else 0.0
            )

            b = self.default_win_loss_ratio if avg_win <= 0 or avg_loss <= 0 else avg_win / avg_loss

        q = 1.0 - p
        raw_kelly = p - (q / b)
        half_kelly = 0.5 * raw_kelly
        return max(0.0, min(half_kelly, 0.50))

    def _compute_qty_and_alloc(
        self,
        price: float,
        signal_confidence: float,
        regime_confidence: float,
        portfolio_equity: float,
        buying_power: float,
        max_trade_dollars: float = 5000.0,
        min_conf_alloc: float = 0.02,
        max_conf_alloc: float = 0.10,
    ) -> tuple[int, float, float, float]:
        if price <= 0 or portfolio_equity <= 0 or buying_power <= 0:
            return 0, 0.0, 0.0, 0.0

        signal_confidence = max(0.0, min(float(signal_confidence), 1.0))
        regime_confidence = max(0.0, min(float(regime_confidence), 1.0))

        blended_conf = 0.7 * signal_confidence + 0.3 * regime_confidence
        confidence_alloc_frac = min_conf_alloc + (max_conf_alloc - min_conf_alloc) * blended_conf

        kelly_frac = self._kelly_fraction()
        effective_kelly = max(0.25, kelly_frac) if kelly_frac > 0 else 0.25

        target_dollars = portfolio_equity * confidence_alloc_frac * effective_kelly
        final_alloc_dollars = min(target_dollars, max_trade_dollars, buying_power)

        qty = int(final_alloc_dollars // price)
        return max(qty, 0), confidence_alloc_frac, kelly_frac, final_alloc_dollars

    def _mark_entry(self, symbol: str, qty: float, price: float, bar_ts: pd.Timestamp) -> None:
        self.open_trades[symbol] = {
            "symbol": symbol,
            "qty": float(qty),
            "entry_price": float(price),
            "entry_timestamp": bar_ts.isoformat(),
        }

    def _mark_exit(self, symbol: str, price: float, bar_ts: pd.Timestamp) -> tuple[float | None, float | None]:
        if symbol not in self.open_trades:
            return None, None

        trade = self.open_trades[symbol]
        qty = float(trade.get("qty", 0.0))
        entry_price = float(trade.get("entry_price", 0.0))
        exit_price = float(price)

        realized_pnl = (exit_price - entry_price) * qty
        notional = entry_price * qty
        realized_pnl_pct = (realized_pnl / notional) if notional > 0 else 0.0

        self.closed_trades.append(
            {
                "symbol": symbol,
                "qty": qty,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_timestamp": trade.get("entry_timestamp", ""),
                "exit_timestamp": bar_ts.isoformat(),
                "realized_pnl": realized_pnl,
                "realized_pnl_pct": realized_pnl_pct,
            }
        )
        del self.open_trades[symbol]
        return realized_pnl, realized_pnl_pct

    def _sync_open_trades_from_broker(self) -> None:
        broker_positions = {p.symbol: p for p in self.broker.get_all_open_positions()}

        for symbol in list(self.open_trades.keys()):
            if symbol not in broker_positions or broker_positions[symbol].qty <= 0:
                del self.open_trades[symbol]

        for symbol, pos in broker_positions.items():
            if symbol not in self.open_trades:
                self.open_trades[symbol] = {
                    "symbol": symbol,
                    "qty": float(pos.qty),
                    "entry_price": float(pos.avg_entry_price),
                    "entry_timestamp": now_utc().isoformat(),
                }

    def _bull_positions(self) -> list[BrokerPosition]:
        return self.broker.get_positions_for_symbols(BULL_UNIVERSE)

    def _bear_positions(self) -> list[BrokerPosition]:
        return self.broker.get_positions_for_symbols(BEAR_UNIVERSE)

    def _close_opposite_regime_positions(self, regime: str, bar_ts: pd.Timestamp) -> None:
        if regime == "BULL":
            opposite = self._bear_positions()
        elif regime == "BEAR":
            opposite = self._bull_positions()
        else:
            opposite = self._bull_positions() + self._bear_positions()

        for pos in opposite:
            latest_price = self._latest_symbol_price(pos.symbol, bar_ts)
            exit_price = latest_price if latest_price is not None else float(pos.avg_entry_price)
            self._mark_exit(pos.symbol, exit_price, bar_ts)
            self.broker.close_symbol_if_open(pos.symbol)

    def _candidate_universe_for_action(self, action_label: str) -> tuple[str, ...]:
        if action_label.startswith("ENTER_TQQQ"):
            return BULL_UNIVERSE
        if action_label.startswith("ENTER_SQQQ"):
            return BEAR_UNIVERSE
        return ()

    def _rank_candidates(self, action_label: str, ts: pd.Timestamp) -> list[dict[str, float | str]]:
        universe = self._candidate_universe_for_action(action_label)
        ranked: list[dict[str, float | str]] = []

        for symbol in universe:
            df = self.live_symbol_frames.get(symbol)
            if df is None or len(df) < 2:
                continue

            sdf = df[df["timestamp"] <= ts].sort_values("timestamp").tail(2)
            if len(sdf) < 2:
                continue

            price = float(sdf["close"].iloc[-1])
            if price < MIN_CANDIDATE_PRICE:
                continue

            mark_pct_chg = self._mark_pct_change(symbol, ts)
            if mark_pct_chg is None:
                continue

            score = float(mark_pct_chg)

            if action_label.startswith("ENTER_TQQQ") and mark_pct_chg < MIN_MARK_PCT_CHG_BULL:
                continue
            if action_label.startswith("ENTER_SQQQ") and mark_pct_chg < MIN_MARK_PCT_CHG_BEAR:
                continue

            ranked.append(
                {
                    "symbol": symbol,
                    "score": score,
                    "mark_pct_chg": float(mark_pct_chg),
                    "price": price,
                }
            )

        ranked.sort(key=lambda x: float(x["score"]), reverse=True)
        return ranked

    def _select_target_symbol(
        self,
        action_label: str,
        ts: pd.Timestamp,
        exclude_symbols: set[str] | None = None,
    ) -> tuple[Optional[str], float, float]:
        exclude_symbols = exclude_symbols or set()
        ranked = self._rank_candidates(action_label, ts)

        if ranked:
            top_preview = ", ".join(
                f"{str(item['symbol'])}:{100.0 * float(item['mark_pct_chg']):.2f}%"
                for item in ranked[:TOP_CANDIDATES_TO_PRINT]
            )
            print(f"[selector] {action_label} | top %chg={top_preview}")

        for item in ranked:
            symbol = str(item["symbol"])
            if symbol in exclude_symbols:
                continue
            return (
                symbol,
                float(item["score"]),
                float(item["mark_pct_chg"]),
            )

        fallback = "TQQQ" if action_label.startswith("ENTER_TQQQ") else "SQQQ"
        if fallback not in exclude_symbols:
            print(f"[selector] {action_label} | no ranked candidates, fallback={fallback}")
            return fallback, 0.0, 0.0

        print(f"[selector] {action_label} | no eligible unheld symbols found")
        return None, 0.0, 0.0

    def _fallback_action_from_regime(self, row: pd.Series, ts: pd.Timestamp) -> str:
        regime = str(row.get("regime", "UNKNOWN"))
        regime_conf = float(row.get("regime_conf", 0.0))

        if regime == "BULL" and regime_conf >= MIN_REGIME_CONFIDENCE:
            symbol, _, pct = self._select_target_symbol("ENTER_TQQQ", ts)
            if symbol and pct >= FALLBACK_BULL_MARK_PCT_CHG:
                return "ENTER_TQQQ"

        if regime == "BEAR" and regime_conf >= MIN_REGIME_CONFIDENCE:
            symbol, _, pct = self._select_target_symbol("ENTER_SQQQ", ts)
            if symbol and pct >= FALLBACK_BEAR_MARK_PCT_CHG:
                return "ENTER_SQQQ"

        return "HOLD"

    def _entry_filter(self, action_label: str, row: pd.Series) -> tuple[bool, str]:
        signal_conf = float(row.get("signal_confidence", 0.0))
        regime_conf = float(row.get("regime_conf", 0.0))
        regime = str(row.get("regime", "UNKNOWN"))

        if signal_conf < MIN_SIGNAL_CONFIDENCE:
            return False, f"blocked_low_signal_confidence:{signal_conf:.3f}"

        if regime_conf < MIN_REGIME_CONFIDENCE:
            return False, f"blocked_low_regime_confidence:{regime_conf:.3f}"

        if self._in_reentry_lockout():
            return False, "blocked_reentry_lockout"

        if "TRANSITION" in action_label and not ALLOW_TRANSITION_ENTRIES:
            return False, "blocked_transition_entry"

        qqq_close = float(row.get("qqq_close", row.get("close", 0.0)) or 0.0)
        if qqq_close <= 0:
            return False, "blocked_invalid_qqq_price"

        regime_ok = (
            (action_label.startswith("ENTER_TQQQ") and regime in {"BULL", "TRANSITION"})
            or (action_label.startswith("ENTER_SQQQ") and regime in {"BEAR", "TRANSITION"})
            or action_label == "HOLD"
        )
        if not regime_ok and action_label != "HOLD":
            return False, f"blocked_regime_mismatch:{regime}"

        return True, "ok"

    def _stop_loss_triggered(
        self,
        current: BrokerPosition,
        ts: pd.Timestamp,
    ) -> tuple[bool, float, float]:
        if current.symbol is None or current.qty <= 0 or current.avg_entry_price <= 0:
            return False, 0.0, 0.0

        current_price = self._latest_symbol_price(current.symbol, ts)
        if current_price is None:
            return False, 0.0, 0.0

        pnl_pct = (current_price - float(current.avg_entry_price)) / float(current.avg_entry_price)
        return pnl_pct <= MAX_LOSS_PNL_PCT, current_price, pnl_pct

    def _apply_stop_losses(self, bar_ts: pd.Timestamp) -> list[dict[str, object]]:
        stop_events: list[dict[str, object]] = []

        for pos in self.broker.get_all_open_positions():
            triggered, stop_price, stop_pnl_pct = self._stop_loss_triggered(pos, bar_ts)
            if not triggered:
                continue

            realized_pnl, realized_pnl_pct = self._mark_exit(pos.symbol, stop_price, bar_ts)
            self.broker.close_symbol_if_open(pos.symbol)

            stop_events.append(
                {
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "price": stop_price,
                    "realized_pnl": realized_pnl,
                    "realized_pnl_pct": realized_pnl_pct if realized_pnl_pct is not None else stop_pnl_pct,
                }
            )

        return stop_events

    def _paper_execute(self, action_label: str, row: pd.Series) -> dict[str, object]:
        bar_ts = pd.to_datetime(row["timestamp"], errors="coerce")
        acct_before = self._record_account_snapshot("before_execution", bar_ts)
        self._sync_open_trades_from_broker()

        bull_before = len(self._bull_positions())
        bear_before = len(self._bear_positions())

        result: dict[str, object] = {
            "message": "hold",
            "selected_symbol": "",
            "selection_side": "",
            "selection_rank_score": "",
            "selection_mark_pct_chg": "",
            "trade_side": "",
            "trade_symbol": "",
            "trade_qty": 0.0,
            "trade_price": 0.0,
            "confidence_alloc_frac": 0.0,
            "kelly_frac": 0.0,
            "final_alloc_dollars": 0.0,
            "entry_price": "",
            "exit_price": "",
            "realized_pnl": "",
            "realized_pnl_pct": "",
            "buying_power_before": float(acct_before["buying_power"]),
            "equity_before": float(acct_before["equity"]),
            "cash_before": float(acct_before["cash"]),
            "buying_power_after": float(acct_before["buying_power"]),
            "equity_after": float(acct_before["equity"]),
            "cash_after": float(acct_before["cash"]),
            "bull_positions_before": bull_before,
            "bear_positions_before": bear_before,
            "bull_positions_after": bull_before,
            "bear_positions_after": bear_before,
        }

        if self.broker.has_open_order_for_symbols(EXEC_SYMBOLS):
            result["message"] = "blocked_open_order"
            return result

        current_regime = str(row.get("regime", "UNKNOWN"))

        if current_regime == "BULL":
            self._close_opposite_regime_positions("BULL", bar_ts)
        elif current_regime == "BEAR":
            self._close_opposite_regime_positions("BEAR", bar_ts)
        elif current_regime == "TRANSITION" and not ALLOW_TRANSITION_ENTRIES:
            self._close_opposite_regime_positions("TRANSITION", bar_ts)

        stop_events = self._apply_stop_losses(bar_ts)
        if stop_events:
            acct_after = self._record_account_snapshot("after_stop_loss", bar_ts)
            self._sync_open_trades_from_broker()
            first = stop_events[0]
            result.update(
                {
                    "message": f"stop_loss_exit_{first['symbol']}",
                    "trade_side": "SELL",
                    "trade_symbol": first["symbol"],
                    "trade_qty": first["qty"],
                    "trade_price": first["price"],
                    "exit_price": first["price"],
                    "realized_pnl": first["realized_pnl"],
                    "realized_pnl_pct": first["realized_pnl_pct"],
                    "buying_power_after": float(acct_after["buying_power"]),
                    "equity_after": float(acct_after["equity"]),
                    "cash_after": float(acct_after["cash"]),
                    "bull_positions_after": len(self._bull_positions()),
                    "bear_positions_after": len(self._bear_positions()),
                }
            )
            return result

        if self._in_cooldown():
            result["message"] = "blocked_cooldown"
            return result

        if action_label == "HOLD":
            result["message"] = "hold"
            return result

        if action_label in {"FORCED_EXIT_TQQQ", "EXIT_TQQQ"}:
            bull_positions = self._bull_positions()
            if bull_positions:
                pos = bull_positions[0]
                latest_price = self._latest_symbol_price(pos.symbol, bar_ts)
                exit_price = latest_price if latest_price is not None else float(pos.avg_entry_price)
                realized_pnl, realized_pnl_pct = self._mark_exit(pos.symbol, exit_price, bar_ts)
                self.broker.close_symbol_if_open(pos.symbol)
                self._mark_order()
                acct_after = self._record_account_snapshot("after_execution", bar_ts)
                self._sync_open_trades_from_broker()

                result.update(
                    {
                        "message": f"closed_{pos.symbol}",
                        "trade_side": "SELL",
                        "trade_symbol": pos.symbol,
                        "trade_qty": pos.qty,
                        "trade_price": exit_price,
                        "exit_price": exit_price,
                        "realized_pnl": realized_pnl if realized_pnl is not None else "",
                        "realized_pnl_pct": realized_pnl_pct if realized_pnl_pct is not None else "",
                        "buying_power_after": float(acct_after["buying_power"]),
                        "equity_after": float(acct_after["equity"]),
                        "cash_after": float(acct_after["cash"]),
                        "bull_positions_after": len(self._bull_positions()),
                        "bear_positions_after": len(self._bear_positions()),
                    }
                )
                return result

            result["message"] = "no_bull_position_to_close"
            return result

        if action_label in {"FORCED_EXIT_SQQQ", "EXIT_SQQQ"}:
            bear_positions = self._bear_positions()
            if bear_positions:
                pos = bear_positions[0]
                latest_price = self._latest_symbol_price(pos.symbol, bar_ts)
                exit_price = latest_price if latest_price is not None else float(pos.avg_entry_price)
                realized_pnl, realized_pnl_pct = self._mark_exit(pos.symbol, exit_price, bar_ts)
                self.broker.close_symbol_if_open(pos.symbol)
                self._mark_order()
                acct_after = self._record_account_snapshot("after_execution", bar_ts)
                self._sync_open_trades_from_broker()

                result.update(
                    {
                        "message": f"closed_{pos.symbol}",
                        "trade_side": "SELL",
                        "trade_symbol": pos.symbol,
                        "trade_qty": pos.qty,
                        "trade_price": exit_price,
                        "exit_price": exit_price,
                        "realized_pnl": realized_pnl if realized_pnl is not None else "",
                        "realized_pnl_pct": realized_pnl_pct if realized_pnl_pct is not None else "",
                        "buying_power_after": float(acct_after["buying_power"]),
                        "equity_after": float(acct_after["equity"]),
                        "cash_after": float(acct_after["cash"]),
                        "bull_positions_after": len(self._bull_positions()),
                        "bear_positions_after": len(self._bear_positions()),
                    }
                )
                return result

            result["message"] = "no_bear_position_to_close"
            return result

        if float(acct_before["buying_power"]) < MIN_BUYING_POWER:
            result["message"] = "blocked_low_buying_power"
            return result

        print(
            f"[entry_check] action={action_label} | "
            f"signal_conf={float(row.get('signal_confidence', 0.0)):.3f} | "
            f"regime={row.get('regime', 'UNKNOWN')} | "
            f"regime_conf={float(row.get('regime_conf', 0.0)):.3f}"
        )

        allowed, block_reason = self._entry_filter(action_label, row)
        if not allowed:
            print(f"[entry_blocked] action={action_label} | reason={block_reason}")
            result["message"] = block_reason
            return result

        if action_label in {"ENTER_TQQQ", "ENTER_TQQQ_TRANSITION"}:
            regime_positions = self._bull_positions()
            if len(regime_positions) >= MAX_BULL_POSITIONS:
                result["message"] = "blocked_max_bull_positions"
                return result

            held_bull_symbols = {p.symbol for p in regime_positions}
            target_symbol, selection_score, selection_mark_pct_chg = self._select_target_symbol(
                action_label,
                bar_ts,
                exclude_symbols=held_bull_symbols,
            )
            if not target_symbol:
                result["message"] = "blocked_no_target_symbol"
                return result

            print(
                f"[entry_target] action={action_label} | "
                f"symbol={target_symbol} | rank_score={selection_score:.6f} | "
                f"pct_chg={100.0 * float(selection_mark_pct_chg):.2f}%"
            )

            price = self._latest_symbol_price(target_symbol, bar_ts)
            if price is None:
                result.update(
                    {
                        "message": f"blocked_missing_price_{target_symbol}",
                        "selected_symbol": target_symbol,
                        "selection_side": "BULL",
                        "selection_rank_score": selection_score,
                        "selection_mark_pct_chg": selection_mark_pct_chg,
                    }
                )
                return result

            qty, conf_alloc, kelly_frac, alloc_dollars = self._compute_qty_and_alloc(
                price=price,
                signal_confidence=float(row["signal_confidence"]),
                regime_confidence=float(row["regime_conf"]),
                portfolio_equity=float(acct_before["equity"]),
                buying_power=float(acct_before["buying_power"]),
                max_trade_dollars=self.target_dollars,
            )

            if current_regime == "TRANSITION":
                qty = max(0, int(qty * TRANSITION_SIZE_MULTIPLIER))
                alloc_dollars *= TRANSITION_SIZE_MULTIPLIER

            print(
                f"[sizing] symbol={target_symbol} | price={price:.2f} | "
                f"equity={float(acct_before['equity']):.2f} | "
                f"buying_power={float(acct_before['buying_power']):.2f} | "
                f"qty={qty} | alloc=${alloc_dollars:.2f} | "
                f"conf_alloc={conf_alloc:.4f} | kelly={kelly_frac:.4f}"
            )

            if qty <= 0:
                print(
                    f"[entry_blocked] action={action_label} | symbol={target_symbol} | "
                    f"reason=blocked_zero_qty"
                )
                result.update(
                    {
                        "message": "blocked_zero_qty",
                        "selected_symbol": target_symbol,
                        "selection_side": "BULL",
                        "selection_rank_score": selection_score,
                        "selection_mark_pct_chg": selection_mark_pct_chg,
                    }
                )
                return result

            self.broker.submit_market_buy(target_symbol, qty)
            self.broker.wait_for_position(target_symbol)
            self._mark_order()
            self._mark_entry(target_symbol, qty, price, bar_ts)
            self._mark_entry_time()

            acct_after = self._record_account_snapshot("after_execution", bar_ts)
            self._sync_open_trades_from_broker()

            result.update(
                {
                    "message": f"entered_{target_symbol}",
                    "selected_symbol": target_symbol,
                    "selection_side": "BULL",
                    "selection_rank_score": selection_score,
                    "selection_mark_pct_chg": selection_mark_pct_chg,
                    "trade_side": "BUY",
                    "trade_symbol": target_symbol,
                    "trade_qty": qty,
                    "trade_price": price,
                    "confidence_alloc_frac": conf_alloc,
                    "kelly_frac": kelly_frac,
                    "final_alloc_dollars": alloc_dollars,
                    "entry_price": price,
                    "buying_power_after": float(acct_after["buying_power"]),
                    "equity_after": float(acct_after["equity"]),
                    "cash_after": float(acct_after["cash"]),
                    "bull_positions_after": len(self._bull_positions()),
                    "bear_positions_after": len(self._bear_positions()),
                }
            )
            return result

        if action_label in {"ENTER_SQQQ", "ENTER_SQQQ_TRANSITION"}:
            regime_positions = self._bear_positions()
            if len(regime_positions) >= MAX_BEAR_POSITIONS:
                result["message"] = "blocked_max_bear_positions"
                return result

            held_bear_symbols = {p.symbol for p in regime_positions}
            target_symbol, selection_score, selection_mark_pct_chg = self._select_target_symbol(
                action_label,
                bar_ts,
                exclude_symbols=held_bear_symbols,
            )
            if not target_symbol:
                result["message"] = "blocked_no_target_symbol"
                return result

            print(
                f"[entry_target] action={action_label} | "
                f"symbol={target_symbol} | rank_score={selection_score:.6f} | "
                f"pct_chg={100.0 * float(selection_mark_pct_chg):.2f}%"
            )

            price = self._latest_symbol_price(target_symbol, bar_ts)
            if price is None:
                result.update(
                    {
                        "message": f"blocked_missing_price_{target_symbol}",
                        "selected_symbol": target_symbol,
                        "selection_side": "BEAR",
                        "selection_rank_score": selection_score,
                        "selection_mark_pct_chg": selection_mark_pct_chg,
                    }
                )
                return result

            qty, conf_alloc, kelly_frac, alloc_dollars = self._compute_qty_and_alloc(
                price=price,
                signal_confidence=float(row["signal_confidence"]),
                regime_confidence=float(row["regime_conf"]),
                portfolio_equity=float(acct_before["equity"]),
                buying_power=float(acct_before["buying_power"]),
                max_trade_dollars=self.target_dollars,
            )

            if current_regime == "TRANSITION":
                qty = max(0, int(qty * TRANSITION_SIZE_MULTIPLIER))
                alloc_dollars *= TRANSITION_SIZE_MULTIPLIER

            print(
                f"[sizing] symbol={target_symbol} | price={price:.2f} | "
                f"equity={float(acct_before['equity']):.2f} | "
                f"buying_power={float(acct_before['buying_power']):.2f} | "
                f"qty={qty} | alloc=${alloc_dollars:.2f} | "
                f"conf_alloc={conf_alloc:.4f} | kelly={kelly_frac:.4f}"
            )

            if qty <= 0:
                print(
                    f"[entry_blocked] action={action_label} | symbol={target_symbol} | "
                    f"reason=blocked_zero_qty"
                )
                result.update(
                    {
                        "message": "blocked_zero_qty",
                        "selected_symbol": target_symbol,
                        "selection_side": "BEAR",
                        "selection_rank_score": selection_score,
                        "selection_mark_pct_chg": selection_mark_pct_chg,
                    }
                )
                return result

            self.broker.submit_market_buy(target_symbol, qty)
            self.broker.wait_for_position(target_symbol)
            self._mark_order()
            self._mark_entry(target_symbol, qty, price, bar_ts)
            self._mark_entry_time()

            acct_after = self._record_account_snapshot("after_execution", bar_ts)
            self._sync_open_trades_from_broker()

            result.update(
                {
                    "message": f"entered_{target_symbol}",
                    "selected_symbol": target_symbol,
                    "selection_side": "BEAR",
                    "selection_rank_score": selection_score,
                    "selection_mark_pct_chg": selection_mark_pct_chg,
                    "trade_side": "BUY",
                    "trade_symbol": target_symbol,
                    "trade_qty": qty,
                    "trade_price": price,
                    "confidence_alloc_frac": conf_alloc,
                    "kelly_frac": kelly_frac,
                    "final_alloc_dollars": alloc_dollars,
                    "entry_price": price,
                    "buying_power_after": float(acct_after["buying_power"]),
                    "equity_after": float(acct_after["equity"]),
                    "cash_after": float(acct_after["cash"]),
                    "bull_positions_after": len(self._bull_positions()),
                    "bear_positions_after": len(self._bear_positions()),
                }
            )
            return result

        result["message"] = "hold"
        return result

    def _log_decision(
        self,
        row: pd.Series,
        action_label: str,
        bull_before: int,
        bear_before: int,
        bull_after: int,
        bear_after: int,
        execution_result: dict[str, object],
    ) -> None:
        bar_ts = pd.to_datetime(row["timestamp"], errors="coerce")
        self.journal.append(
            {
                "timestamp_utc": now_utc().isoformat(),
                "timestamp_et": pd.Timestamp.now(tz=TIMEZONE).isoformat(),
                "bar_timestamp_et": "" if pd.isna(bar_ts) else bar_ts.isoformat(),
                "regime": str(row["regime"]),
                "regime_conf": float(row["regime_conf"]),
                "signal_confidence": float(row["signal_confidence"]),
                "bull_score": float(row["bull_score"]),
                "bear_score": float(row["bear_score"]),
                "action_label": action_label,
                "selected_symbol": execution_result.get("selected_symbol", ""),
                "selection_side": execution_result.get("selection_side", ""),
                "selection_rank_score": execution_result.get("selection_rank_score", ""),
                "selection_mark_pct_chg": execution_result.get("selection_mark_pct_chg", ""),
                "bull_positions_before": bull_before,
                "bear_positions_before": bear_before,
                "bull_positions_after": bull_after,
                "bear_positions_after": bear_after,
                "qqq_close": float(row["qqq_close"]),
                "buying_power_before": execution_result.get("buying_power_before", ""),
                "equity_before": execution_result.get("equity_before", ""),
                "cash_before": execution_result.get("cash_before", ""),
                "buying_power_after": execution_result.get("buying_power_after", ""),
                "equity_after": execution_result.get("equity_after", ""),
                "cash_after": execution_result.get("cash_after", ""),
                "trade_side": execution_result.get("trade_side", ""),
                "trade_symbol": execution_result.get("trade_symbol", ""),
                "trade_qty": execution_result.get("trade_qty", ""),
                "trade_price": execution_result.get("trade_price", ""),
                "confidence_alloc_frac": execution_result.get("confidence_alloc_frac", ""),
                "kelly_frac": execution_result.get("kelly_frac", ""),
                "final_alloc_dollars": execution_result.get("final_alloc_dollars", ""),
                "entry_price": execution_result.get("entry_price", ""),
                "exit_price": execution_result.get("exit_price", ""),
                "realized_pnl": execution_result.get("realized_pnl", ""),
                "realized_pnl_pct": execution_result.get("realized_pnl_pct", ""),
                "message": execution_result.get("message", ""),
            }
        )

    def _seed_live_buffers(self) -> None:
        hist = self.broker.fetch_recent_symbol_bars(list(ALL_SYMBOLS), WARMUP_BARS)

        for symbol in ALL_SYMBOLS:
            sdf = hist[hist["symbol"] == symbol].copy()
            if len(sdf) == 0:
                continue

            frame = pd.DataFrame(
                {
                    "timestamp": sdf["timestamp"],
                    "open": sdf["open"],
                    "high": sdf["high"],
                    "low": sdf["low"],
                    "close": sdf["close"],
                    "volume": sdf["volume"],
                }
            )
            frame = (
                frame.dropna(subset=["timestamp", "close"])
                .drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            self.live_symbol_frames[symbol] = frame.tail(WARMUP_BARS).reset_index(drop=True)

        feature_sizes = {s: len(self.live_symbol_frames[s]) for s in FEATURE_SYMBOLS}
        print(f"[paper] seeded live buffers | feature_sizes={feature_sizes}")

    def _upsert_live_bar(self, symbol: str, row: dict[str, object]) -> None:
        df = self.live_symbol_frames[symbol]
        new_row = pd.DataFrame([row])

        if len(df) > 0 and (df["timestamp"] == row["timestamp"]).any():
            df = df[df["timestamp"] != row["timestamp"]]

        df = pd.concat([df, new_row], ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        df = df.tail(WARMUP_BARS).reset_index(drop=True)
        self.live_symbol_frames[symbol] = df

    def _latest_common_timestamp(self) -> Optional[pd.Timestamp]:
        latest_times: list[pd.Timestamp] = []

        for symbol in FEATURE_SYMBOLS:
            df = self.live_symbol_frames.get(symbol)
            if df is None or len(df) == 0:
                return None

            latest = pd.to_datetime(df["timestamp"].iloc[-1], errors="coerce")
            if pd.isna(latest):
                return None
            latest_times.append(latest)

        candidate = min(latest_times)
        if pd.isna(candidate):
            return None

        for symbol in FEATURE_SYMBOLS:
            df = self.live_symbol_frames.get(symbol)
            if not (df["timestamp"] == candidate).any():
                return None

        return candidate

    def _build_merged_from_live_buffers(self, ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        frames: list[pd.DataFrame] = []

        for symbol in FEATURE_SYMBOLS:
            df = self.live_symbol_frames.get(symbol)
            if df is None or len(df) == 0:
                return None

            symbol_rows = df[df["timestamp"] <= ts].tail(WARMUP_BARS).copy()
            if len(symbol_rows) == 0:
                return None

            renamed = symbol_rows.rename(
                columns={
                    "open": f"{symbol.lower()}_open",
                    "high": f"{symbol.lower()}_high",
                    "low": f"{symbol.lower()}_low",
                    "close": f"{symbol.lower()}_close",
                    "volume": f"{symbol.lower()}_volume",
                }
            )

            frames.append(
                renamed[
                    [
                        "timestamp",
                        f"{symbol.lower()}_open",
                        f"{symbol.lower()}_high",
                        f"{symbol.lower()}_low",
                        f"{symbol.lower()}_close",
                        f"{symbol.lower()}_volume",
                    ]
                ]
            )

        merged = frames[0]
        for frame in frames[1:]:
            merged = pd.merge(merged, frame, on="timestamp", how="inner")

        if len(merged) == 0:
            return None

        merged = merged.sort_values("timestamp").tail(WARMUP_BARS).reset_index(drop=True)
        merged["open"] = merged["qqq_open"]
        merged["high"] = merged["qqq_high"]
        merged["low"] = merged["qqq_low"]
        merged["close"] = merged["qqq_close"]
        merged["volume"] = merged["qqq_volume"]

        if not (merged["timestamp"] == ts).any():
            return None

        return merged

    def _print_heartbeat(self, latest_ts: pd.Timestamp) -> None:
        now_ts = time.time()
        if (now_ts - self.last_heartbeat_time) < HEARTBEAT_SECONDS:
            return

        feature_sizes = {symbol: len(self.live_symbol_frames[symbol]) for symbol in FEATURE_SYMBOLS}
        equity_text = "n/a" if self.last_equity is None else f"${self.last_equity:,.2f}"
        print(
            f"[stream] heartbeat | latest_common_ts={latest_ts} | "
            f"feature_buffers={feature_sizes} | equity={equity_text} | "
            f"bull_positions={len(self._bull_positions())} | bear_positions={len(self._bear_positions())} | "
            f"closed_trades={len(self.closed_trades)}"
        )
        self.last_heartbeat_time = now_ts

    def _process_completed_timestamp(self, ts: pd.Timestamp) -> None:
        if self.last_processed_timestamp is not None and ts <= self.last_processed_timestamp:
            return

        merged = self._build_merged_from_live_buffers(ts)
        if merged is None or len(merged) == 0:
            print("[paper] merged feature frame is empty before build_features")
            return

        print(
            f"[paper] merged rows before features={len(merged)} | "
            f"non_null qqq={merged['qqq_close'].notna().sum()} | "
            f"non_null tqqq={merged['tqqq_close'].notna().sum()} | "
            f"non_null sqqq={merged['sqqq_close'].notna().sum()}"
        )

        feat = self.strategy.build_features(merged)
        self.strategy._load_vec_norm(feat)

        if len(feat) == 0:
            print("[paper] no feature rows available after live merge")
            print(
                "[paper] last merged timestamps:\n"
                f"{merged[['timestamp', 'qqq_close', 'tqqq_close', 'sqqq_close']].tail(5)}"
            )
            return

        row = feat.iloc[-1]
        bar_ts = pd.to_datetime(row["timestamp"], errors="coerce")
        if pd.isna(bar_ts):
            print("[paper] invalid latest timestamp")
            return

        if not self._is_regular_hours_bar(bar_ts):
            self.last_processed_timestamp = ts
            print(f"[paper] skipping non-regular-hours bar: {bar_ts}")
            return

        bull_before = len(self._bull_positions())
        bear_before = len(self._bear_positions())

        self.strategy.update_position_state(row)
        obs = self.strategy.get_obs(row)
        action = self.strategy.select_action(obs)
        raw_action_label = self.strategy.execute_action(action, row)
        action_label = self._desired_action_from_signal(raw_action_label)

        print(
            f"[signal] raw_action={action} | raw_label={raw_action_label} | "
            f"mapped_label={action_label} | regime={row.get('regime', 'UNKNOWN')} | "
            f"regime_conf={float(row.get('regime_conf', 0.0)):.3f} | "
            f"signal_conf={float(row.get('signal_confidence', 0.0)):.3f}"
        )

        if action_label == "HOLD":
            fallback_label = self._fallback_action_from_regime(row, bar_ts)
            if fallback_label != "HOLD":
                print(f"[fallback] replacing HOLD with {fallback_label}")
                action_label = fallback_label

        execution_result = self._paper_execute(action_label, row)

        bull_after = len(self._bull_positions())
        bear_after = len(self._bear_positions())

        self._log_decision(
            row=row,
            action_label=action_label,
            bull_before=bull_before,
            bear_before=bear_before,
            bull_after=bull_after,
            bear_after=bear_after,
            execution_result=execution_result,
        )

        self.last_processed_timestamp = ts
        self._write_account_equity_curve()
        self._print_heartbeat(ts)

        message = str(execution_result.get("message", ""))
        trade_side = str(execution_result.get("trade_side", ""))
        trade_symbol = str(execution_result.get("trade_symbol", ""))
        trade_qty = execution_result.get("trade_qty", 0)
        trade_price = execution_result.get("trade_price", 0.0)
        kelly_frac = execution_result.get("kelly_frac", 0.0)
        conf_alloc = execution_result.get("confidence_alloc_frac", 0.0)
        alloc_dollars = execution_result.get("final_alloc_dollars", 0.0)
        realized_pnl = execution_result.get("realized_pnl", "")
        realized_pnl_pct = execution_result.get("realized_pnl_pct", "")

        equity_before = execution_result.get("equity_before", 0.0)
        equity_after = execution_result.get("equity_after", 0.0)
        cash_after = execution_result.get("cash_after", 0.0)
        buying_power_after = execution_result.get("buying_power_after", 0.0)

        selected_symbol = execution_result.get("selected_symbol", "")
        selection_side = execution_result.get("selection_side", "")
        selection_score = execution_result.get("selection_rank_score", "")
        selection_mark_pct_chg = execution_result.get("selection_mark_pct_chg", "")

        print(
            f"[paper] {bar_ts} | regime={row['regime']} | "
            f"regime_conf={float(row['regime_conf']):.2f} | "
            f"signal_conf={float(row['signal_confidence']):.2f} | "
            f"signal={action_label} | exec={message} | "
            f"bull_before={bull_before} | bear_before={bear_before} | "
            f"bull_after={bull_after} | bear_after={bear_after}"
        )

        print(
            f"[ACCOUNT] equity_before=${float(equity_before):.2f} | "
            f"equity_after=${float(equity_after):.2f} | "
            f"cash_after=${float(cash_after):.2f} | "
            f"buying_power_after=${float(buying_power_after):.2f}"
        )

        if selected_symbol:
            try:
                score_txt = f"{float(selection_score):.4f}"
            except Exception:
                score_txt = str(selection_score)

            try:
                mark_txt = f"{100.0 * float(selection_mark_pct_chg):.2f}%"
            except Exception:
                mark_txt = str(selection_mark_pct_chg)

            print(
                f"[SELECT] side={selection_side} | symbol={selected_symbol} | "
                f"score={score_txt} | %chg={mark_txt}"
            )

        if trade_side and trade_symbol:
            print(
                f"[TRADE] {trade_side} {trade_symbol} | "
                f"qty={int(float(trade_qty))} | "
                f"price={float(trade_price):.2f} | "
                f"conf_alloc={float(conf_alloc):.4f} | "
                f"kelly={float(kelly_frac):.4f} | "
                f"alloc=${float(alloc_dollars):.2f}"
            )

        if realized_pnl != "":
            print(
                f"[PNL] {trade_symbol} | "
                f"realized=${float(realized_pnl):.2f} | "
                f"realized_pct={100.0 * float(realized_pnl_pct):.2f}% | "
                f"closed_trades={len(self.closed_trades)}"
            )

    async def _on_bar(self, bar) -> None:
        try:
            ts = pd.to_datetime(getattr(bar, "timestamp", None), utc=True, errors="coerce")
            if pd.isna(ts):
                return

            ts = ts.tz_convert(TIMEZONE).floor("min")
            symbol = str(getattr(bar, "symbol", "")).upper()
            if symbol not in ALL_SYMBOLS:
                return

            row = {
                "timestamp": ts,
                "open": float(getattr(bar, "open")),
                "high": float(getattr(bar, "high")),
                "low": float(getattr(bar, "low")),
                "close": float(getattr(bar, "close")),
                "volume": float(getattr(bar, "volume")),
            }

            self._upsert_live_bar(symbol, row)
            common_ts = self._latest_common_timestamp()
            if common_ts is not None:
                self._process_completed_timestamp(common_ts)

        except Exception as exc:
            print(f"[stream] handler error: {exc}", file=sys.stderr)

    def _run_stream_once(self) -> None:
        self._seed_live_buffers()
        self._sync_open_trades_from_broker()

        stream = self.broker.create_stream()
        stream.subscribe_bars(self._on_bar, *ALL_SYMBOLS)

        print("[stream] subscribed to live bars")
        print(f"[stream] symbols={','.join(ALL_SYMBOLS)}")

        stream.run()

    def run_forever(self) -> None:
        print("[paper] streaming paper trader starting...")
        print(f"[paper] journal={JOURNAL_PATH}")
        print(f"[paper] equity_curve={EQUITY_CURVE_PATH}")

        while True:
            try:
                self._run_stream_once()
            except KeyboardInterrupt:
                print("\n[paper] stopped by user")
                return
            except Exception as exc:
                print(f"[stream] reconnecting after error: {exc}", file=sys.stderr)
                self.last_processed_timestamp = None
                time.sleep(3)


if __name__ == "__main__":
    trader = ProductionPaperTrader()
    trader.run_forever()