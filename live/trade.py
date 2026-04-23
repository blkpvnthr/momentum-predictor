from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderStatus, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from portfolio_rebalancer import AlphaPortfolioConfig, AlphaPortfolioOptimizer
from trading_env import TradeCentricMDPConfig, TradingEnv
from trading_system import LiveTrader


THIS_FILE = Path(__file__).resolve()
LIVE_DIR = THIS_FILE.parent
PROJECT_ROOT = LIVE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

LOG_DIR = LIVE_DIR / "logs"
JOURNAL_PATH = LOG_DIR / "paper_trader_journal.csv"
EQUITY_CURVE_PATH = LOG_DIR / "paper_trader_equity_curve.csv"

TIMEZONE = "America/New_York"
BAR_TIMEFRAME = TimeFrame.Minute
WARMUP_BARS = 500

CORE_SYMBOLS = ["QQQ", "TQQQ", "SQQQ"]
UNIVERSE_SYMBOLS = [
    "SPY", "RGTI", "INTC", "XLK", "XLF", "XLE", "XLV", "SOXX", "APLD", "CIFR", "VIXY",
    "TQQQ", "SQQQ", "SOXL", "SOXS", "TECL", "SPXL", "SPXU", "SOFI", "TTD", "SMCI", "CORZ",
    "TSLL", "IONQ", "QBTS", "QUBT", "IBIT", "LUNR", "IREN", "TSLG"
]
ALL_SYMBOLS = tuple(sorted(set(CORE_SYMBOLS + UNIVERSE_SYMBOLS)))

POLL_SECONDS = 30
MIN_BUYING_POWER = 250.0
ORDER_WAIT_SECONDS = 20
COOLDOWN_SECONDS = 8
MIN_NOTIONAL_TO_TRADE = 100.0
MIN_WEIGHT_DELTA = 0.0075

# Max-return execution knobs
TOP_CANDIDATES_FOR_ALLOCATION = 8
TOP_CANDIDATES_FOR_NEW_BUYS = 5
MAX_SELL_FRACTION_PER_CYCLE = 1.00
BASE_MAX_BUY_FRACTION_PER_CYCLE = 0.90
MAX_BUY_FRACTION_CAP = 1.00
MAX_NEW_POSITIONS_PER_CYCLE = 8
FULL_EXIT_WEIGHT_THRESHOLD = 0.01

VOL_LOOKBACK_BARS = 30
VOL_TARGET = 0.0015
VOL_SCALE_MIN = 0.85
VOL_SCALE_MAX = 1.50

HEARTBEAT_SECONDS = 60


@dataclass
class BrokerPosition:
    symbol: Optional[str] = None
    qty: float = 0.0
    market_value: float = 0.0
    avg_entry_price: float = 0.0


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def build_env_config(max_episode_steps: int = 256) -> TradeCentricMDPConfig:
    return TradeCentricMDPConfig(
        initial_cash=100_000.0,
        hmax=100,
        transaction_cost_pct=0.001,
        invalid_action_penalty=0.001,
        turbulence_threshold_quantile=0.99,
        max_episode_steps=max_episode_steps,
        allow_fractional_clip_to_cash=True,
        reward_scale=1.0,
        min_feature_lookback=30,
        target_num_stocks=30,
        hold_winner_bonus_weight=0.20,
        strong_trend_adx_threshold=20.0,
        strong_trend_macd_floor=0.0,
        loser_hold_penalty_weight=0.15,
        loser_hold_threshold=0.005,
        stagnation_penalty_weight=0.002,
        stagnation_threshold=0.002,
        stagnation_bars_threshold=8,
        small_exit_penalty=0.005,
        small_exit_threshold=0.0075,
        premature_exit_penalty_weight=0.75,
        premature_exit_lookahead=5,
        premature_exit_min_future_gain=0.0075,
        trade_reward_weight=1.00,
        positive_trade_exponent=1.25,
        negative_trade_linear_weight=1.25,
        mfe_capture_bonus_weight=0.90,
        min_bars_for_capture_bonus=3,
        no_position_cash_idle_penalty=0.0,
    )


class CsvJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = [
            "timestamp_utc",
            "bar_timestamp_et",
            "equity_before",
            "equity_after",
            "cash_before",
            "cash_after",
            "buying_power_before",
            "buying_power_after",
            "portfolio_value_model",
            "signal_confidence",
            "regime",
            "regime_conf",
            "turbulence",
            "turbulence_threshold",
            "realized_volatility",
            "execution_scale",
            "optimizer_expected_return_annual",
            "optimizer_expected_vol_annual",
            "optimizer_expected_sharpe",
            "optimizer_weight_sum",
            "optimizer_symbol_count",
            "optimizer_portfolio_alpha",
            "optimizer_gross_exposure",
            "optimizer_per_asset_cap",
            "optimizer_risk_aversion",
            "trade_symbol",
            "trade_side",
            "trade_qty",
            "trade_price",
            "target_weight",
            "current_weight",
            "delta_weight",
            "action_value",
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

    def get_account_snapshot(self) -> dict[str, float]:
        acct = self.trading.get_account()
        return {
            "buying_power": float(acct.buying_power),
            "equity": float(acct.equity),
            "cash": float(acct.cash),
        }

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

    def list_open_orders(self) -> list:
        try:
            return self.trading.get_orders()
        except Exception:
            return []

    def has_open_order_for_symbol(self, symbol: str) -> bool:
        open_statuses = {
            OrderStatus.NEW,
            OrderStatus.ACCEPTED,
            OrderStatus.PENDING_NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.ACCEPTED_FOR_BIDDING,
            OrderStatus.CALCULATED,
        }
        for order in self.list_open_orders():
            try:
                if order.symbol == symbol and order.status in open_statuses:
                    return True
            except Exception:
                continue
        return False

    def get_position(self, symbol: str) -> BrokerPosition:
        for p in self.get_all_open_positions():
            if p.symbol == symbol:
                return p
        return BrokerPosition()

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

    def wait_until_qty_at_or_below(self, symbol: str, target_qty: float, timeout_seconds: int = ORDER_WAIT_SECONDS) -> None:
        started = time.time()
        while time.time() - started < timeout_seconds:
            if self.get_position(symbol).qty <= target_qty:
                return
            time.sleep(0.5)
        raise TimeoutError(f"{symbol} position did not reach target qty <= {target_qty} within timeout")

    def wait_for_position(self, symbol: str, timeout_seconds: int = ORDER_WAIT_SECONDS) -> BrokerPosition:
        started = time.time()
        while time.time() - started < timeout_seconds:
            pos = self.get_position(symbol)
            if pos.qty > 0:
                return pos
            time.sleep(0.5)
        return BrokerPosition()

    def fetch_recent_symbol_bars(
        self,
        symbols: list[str],
        lookback_bars: int = WARMUP_BARS,
    ) -> pd.DataFrame:
        end_ts = now_utc()
        start_ts = end_ts - timedelta(days=15)

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
    def __init__(self):
        self.journal = CsvJournal(JOURNAL_PATH)
        self.broker = AlpacaPaperBroker(ENV_PATH)
        self.strategy = LiveTrader()
        self.strategy.env_config = build_env_config(max_episode_steps=256)

        self.optimizer = AlphaPortfolioOptimizer(
            AlphaPortfolioConfig(
                lookback_bars=180,
                min_history_rows=60,
                risk_free_rate_annual=0.04,
                cash_buffer_weight=0.01,
                model_weight_power=1.35,
                recent_momentum_bars=20,
                medium_momentum_bars=60,
                alpha_model_weight=0.65,
                alpha_recent_momentum_weight=0.25,
                alpha_medium_momentum_weight=0.10,
                base_per_asset_weight_cap=0.35,
                max_per_asset_weight_cap=0.60,
                base_gross_exposure=0.96,
                max_gross_exposure=0.99,
                top_k=5,
                top_k_min_weight_share=0.75,
                shrinkage=0.15,
                turnover_penalty=0.0005,
                temperature=0.45,
                max_iter=3000,
                step_size=0.08,
            )
        )

        self.last_order_time: float = 0.0
        self.last_heartbeat_time: float = 0.0
        self.last_processed_timestamp: Optional[pd.Timestamp] = None

        self.account_history: list[dict[str, float | str]] = []
        self.last_equity: float | None = None
        self.live_symbol_frames: dict[str, pd.DataFrame] = {
            symbol: pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            for symbol in ALL_SYMBOLS
        }

    def _in_cooldown(self) -> bool:
        return (time.time() - self.last_order_time) < COOLDOWN_SECONDS

    def _mark_order(self) -> None:
        self.last_order_time = time.time()

    def _is_regular_hours_bar(self, ts: pd.Timestamp) -> bool:
        if ts.tzinfo is None:
            return False
        hhmm = ts.hour * 100 + ts.minute
        return 930 <= hhmm <= 1600

    def _broker_held_symbols(self) -> list[str]:
        held = []
        for pos in self.broker.get_all_open_positions():
            symbol = str(pos.symbol) if pos.symbol is not None else ""
            if symbol and float(pos.qty) > 0:
                held.append(symbol)
        return sorted(set(held))

    def _record_account_snapshot(self, label: str, bar_ts: pd.Timestamp | None = None) -> dict[str, float | str]:
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

        preview_sizes = {k: len(v) for k, v in list(self.live_symbol_frames.items())[:3]}
        print(
            f"[paper] seeded live buffers | symbol_count={len(ALL_SYMBOLS)} | "
            f"buffer_sizes={preview_sizes}"
        )

    def _rebuild_from_poll(self) -> pd.Timestamp:
        hist = self.broker.fetch_recent_symbol_bars(list(ALL_SYMBOLS), WARMUP_BARS)
        latest_ts: Optional[pd.Timestamp] = None

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
            ts = pd.to_datetime(frame["timestamp"].iloc[-1], errors="coerce")
            if latest_ts is None or ts < latest_ts:
                latest_ts = ts

        if latest_ts is None:
            raise RuntimeError("No latest timestamp available from polled bars")

        return latest_ts

    def _build_merged_from_live_buffers(self, ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        qqq_df = self.live_symbol_frames.get("QQQ")
        if qqq_df is None or len(qqq_df) == 0:
            return None

        base = (
            qqq_df[qqq_df["timestamp"] <= ts]
            .tail(WARMUP_BARS)
            .copy()
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )
        if len(base) == 0:
            return None

        merged = base.rename(
            columns={
                "open": "qqq_open",
                "high": "qqq_high",
                "low": "qqq_low",
                "close": "qqq_close",
                "volume": "qqq_volume",
            }
        )[
            ["timestamp", "qqq_open", "qqq_high", "qqq_low", "qqq_close", "qqq_volume"]
        ].copy()

        for symbol in ALL_SYMBOLS:
            if symbol == "QQQ":
                continue

            df = self.live_symbol_frames.get(symbol)
            if df is None or len(df) == 0:
                continue

            symbol_rows = (
                df[df["timestamp"] <= ts]
                .tail(WARMUP_BARS)
                .copy()
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="last")
            )
            if len(symbol_rows) == 0:
                continue

            renamed = symbol_rows.rename(
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

            merged = merged.merge(renamed, on="timestamp", how="left")

        merged = merged.sort_values("timestamp").ffill().bfill()
        merged = merged.dropna(subset=["qqq_open", "qqq_high", "qqq_low", "qqq_close", "qqq_volume"])

        merged["open"] = merged["qqq_open"]
        merged["high"] = merged["qqq_high"]
        merged["low"] = merged["qqq_low"]
        merged["close"] = merged["qqq_close"]
        merged["volume"] = merged["qqq_volume"]

        if len(merged) == 0:
            return None

        return merged.tail(WARMUP_BARS).reset_index(drop=True)

    def _current_weights(self, symbols: list[str], equity: float) -> dict[str, float]:
        positions = {p.symbol: p for p in self.broker.get_all_open_positions()}
        out: dict[str, float] = {}
        for symbol in symbols:
            pos = positions.get(symbol)
            mv = float(pos.market_value) if pos is not None else 0.0
            out[symbol] = (mv / equity) if equity > 0 else 0.0
        return out

    def _target_weights_from_action(
        self,
        action: np.ndarray,
        stock_symbols: list[str],
        regime: str,
        regime_conf: float,
        signal_confidence: float,
    ) -> dict[str, float]:
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        positive = np.clip(action, 0.0, 1.0)

        if positive.sum() <= 1e-8:
            return {s: 0.0 for s in stock_symbols}

        regime = str(regime or "").upper()
        regime_conf = float(np.clip(regime_conf, 0.0, 1.0))
        signal_confidence = float(np.clip(signal_confidence, 0.0, 1.0))

        if regime == "BULL":
            power = 1.60 + 0.60 * regime_conf + 0.40 * signal_confidence
        elif regime == "TRANSITION":
            power = 1.25 + 0.25 * signal_confidence
        else:
            power = 1.05

        scaled = np.power(positive, power)
        if scaled.sum() <= 1e-8:
            scaled = positive

        weights = scaled / scaled.sum()
        return {s: float(w) for s, w in zip(stock_symbols, weights)}

    def _build_price_history_for_optimizer(
        self,
        symbols: list[str],
        ts: pd.Timestamp,
        lookback_bars: int,
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for symbol in symbols:
            df = self.live_symbol_frames.get(symbol)
            if df is None or len(df) == 0:
                continue

            sdf = (
                df[df["timestamp"] <= ts]
                .tail(lookback_bars)
                .copy()
                .sort_values("timestamp")
            )
            if len(sdf) == 0:
                continue

            temp = sdf[["timestamp", "close"]].copy()
            temp["symbol"] = symbol
            rows.append(temp[["timestamp", "symbol", "close"]])

        if not rows:
            return pd.DataFrame(columns=["timestamp", "symbol", "close"])

        out = pd.concat(rows, ignore_index=True)
        out = out.dropna(subset=["timestamp", "symbol", "close"])
        out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        return out

    def _candidate_execution_universe(
        self,
        stock_symbols: list[str],
        model_target_weights: dict[str, float],
        held_symbols: list[str],
    ) -> list[str]:
        ranked_model_symbols = sorted(
            stock_symbols,
            key=lambda s: model_target_weights.get(s, 0.0),
            reverse=True,
        )

        top_model_symbols = [
            s for s in ranked_model_symbols
            if model_target_weights.get(s, 0.0) > 0.0
        ][:TOP_CANDIDATES_FOR_ALLOCATION]

        return sorted(set(top_model_symbols) | set(held_symbols))

    def _optimizer_target_weights(
        self,
        ts: pd.Timestamp,
        stock_symbols: list[str],
        action: np.ndarray,
        equity: float,
        regime: str,
        regime_conf: float,
        signal_confidence: float,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        held_symbols = self._broker_held_symbols()

        model_target_weights = self._target_weights_from_action(
            action=action,
            stock_symbols=stock_symbols,
            regime=regime,
            regime_conf=regime_conf,
            signal_confidence=signal_confidence,
        )

        optimizer_universe = self._candidate_execution_universe(
            stock_symbols=stock_symbols,
            model_target_weights=model_target_weights,
            held_symbols=held_symbols,
        )

        if not optimizer_universe:
            return {}, {}, {
                "optimizer_expected_return_annual": 0.0,
                "optimizer_expected_vol_annual": 0.0,
                "optimizer_expected_sharpe": 0.0,
                "optimizer_weight_sum": 0.0,
                "optimizer_symbol_count": 0.0,
                "optimizer_portfolio_alpha": 0.0,
                "optimizer_gross_exposure": 0.0,
                "optimizer_per_asset_cap": 0.0,
                "optimizer_risk_aversion": 0.0,
            }

        current_weights = self._current_weights(optimizer_universe, equity)
        price_history = self._build_price_history_for_optimizer(
            symbols=optimizer_universe,
            ts=ts,
            lookback_bars=self.optimizer.config.lookback_bars,
        )

        if len(price_history) == 0:
            return model_target_weights, current_weights, {
                "optimizer_expected_return_annual": 0.0,
                "optimizer_expected_vol_annual": 0.0,
                "optimizer_expected_sharpe": 0.0,
                "optimizer_weight_sum": float(sum(model_target_weights.values())),
                "optimizer_symbol_count": float(len(optimizer_universe)),
                "optimizer_portfolio_alpha": 0.0,
                "optimizer_gross_exposure": 0.0,
                "optimizer_per_asset_cap": 0.0,
                "optimizer_risk_aversion": 0.0,
            }

        try:
            optimized_weights, diag = self.optimizer.optimize(
                price_history=price_history,
                tradable_symbols=optimizer_universe,
                current_weights=current_weights,
                model_target_weights=model_target_weights,
                regime=regime,
                regime_conf=regime_conf,
                signal_confidence=signal_confidence,
            )
            return optimized_weights, current_weights, diag
        except Exception as exc:
            print(f"[optimizer] fallback to model weights: {exc}")
            return model_target_weights, current_weights, {
                "optimizer_expected_return_annual": 0.0,
                "optimizer_expected_vol_annual": 0.0,
                "optimizer_expected_sharpe": 0.0,
                "optimizer_weight_sum": float(sum(model_target_weights.values())),
                "optimizer_symbol_count": float(len(optimizer_universe)),
                "optimizer_portfolio_alpha": 0.0,
                "optimizer_gross_exposure": 0.0,
                "optimizer_per_asset_cap": 0.0,
                "optimizer_risk_aversion": 0.0,
            }

    def _estimate_market_volatility(self, ts: pd.Timestamp) -> float:
        qqq_df = self.live_symbol_frames.get("QQQ")
        if qqq_df is None or len(qqq_df) < (VOL_LOOKBACK_BARS + 2):
            return 0.0

        recent = (
            qqq_df[qqq_df["timestamp"] <= ts]
            .tail(VOL_LOOKBACK_BARS + 1)
            .copy()
            .sort_values("timestamp")
        )
        if len(recent) < 5:
            return 0.0

        returns = recent["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < 5:
            return 0.0

        return float(returns.std(ddof=0))

    def _volatility_execution_scale(self, realized_volatility: float, regime: str, regime_conf: float, signal_confidence: float) -> float:
        if not np.isfinite(realized_volatility) or realized_volatility <= 0:
            return 1.0

        raw_scale = VOL_TARGET / max(realized_volatility, 1e-8)
        regime = str(regime or "").upper()
        regime_conf = float(np.clip(regime_conf, 0.0, 1.0))
        signal_confidence = float(np.clip(signal_confidence, 0.0, 1.0))

        # In strong bullish, high-confidence conditions, do not choke risk-taking.
        if regime == "BULL":
            floor = 1.0 + 0.20 * regime_conf + 0.15 * signal_confidence
            return float(np.clip(max(raw_scale, floor), 1.0, VOL_SCALE_MAX))

        if regime == "TRANSITION":
            return float(np.clip(raw_scale, 0.90, 1.10))

        return float(np.clip(raw_scale, VOL_SCALE_MIN, 1.15))

    def _execution_aggressiveness(self, regime: str, regime_conf: float, signal_confidence: float) -> tuple[float, int]:
        regime = str(regime or "").upper()
        regime_conf = float(np.clip(regime_conf, 0.0, 1.0))
        signal_confidence = float(np.clip(signal_confidence, 0.0, 1.0))

        if regime == "BULL":
            buy_fraction = min(MAX_BUY_FRACTION_CAP, BASE_MAX_BUY_FRACTION_PER_CYCLE + 0.10 * regime_conf + 0.10 * signal_confidence)
            max_new = int(round(MAX_NEW_POSITIONS_PER_CYCLE + 2 * regime_conf + 2 * signal_confidence))
            return buy_fraction, max_new

        if regime == "TRANSITION":
            return 0.70, max(3, MAX_NEW_POSITIONS_PER_CYCLE - 2)

        return 0.50, max(2, MAX_NEW_POSITIONS_PER_CYCLE - 4)

    def _place_rebalance_orders(
        self,
        ts: pd.Timestamp,
        stock_symbols: list[str],
        action: np.ndarray,
        row: pd.Series,
    ) -> list[dict[str, object]]:
        acct_before = self._record_account_snapshot("before_execution", ts)
        equity_before = float(acct_before["equity"])
        cash_before = float(acct_before["cash"])
        buying_power_before = float(acct_before["buying_power"])

        regime = str(row.get("regime", ""))
        regime_conf = float(row.get("regime_conf", 0.0))
        signal_confidence = float(row.get("signal_confidence", 0.0))

        target_weights, current_weights, optimizer_diag = self._optimizer_target_weights(
            ts=ts,
            stock_symbols=stock_symbols,
            action=action,
            equity=equity_before,
            regime=regime,
            regime_conf=regime_conf,
            signal_confidence=signal_confidence,
        )

        execution_symbols = sorted(set(target_weights.keys()) | set(current_weights.keys()))
        realized_volatility = self._estimate_market_volatility(ts)
        execution_scale = self._volatility_execution_scale(
            realized_volatility=realized_volatility,
            regime=regime,
            regime_conf=regime_conf,
            signal_confidence=signal_confidence,
        )
        buy_fraction, max_new_positions = self._execution_aggressiveness(
            regime=regime,
            regime_conf=regime_conf,
            signal_confidence=signal_confidence,
        )

        if self._in_cooldown():
            return [{
                "message": "blocked_cooldown",
                "equity_before": equity_before,
                "equity_after": equity_before,
                "cash_before": cash_before,
                "cash_after": cash_before,
                "buying_power_before": buying_power_before,
                "buying_power_after": buying_power_before,
                "realized_volatility": realized_volatility,
                "execution_scale": execution_scale,
                **optimizer_diag,
            }]

        prices = {}
        for symbol in execution_symbols:
            df = self.live_symbol_frames.get(symbol)
            if df is None or len(df) == 0:
                continue
            sdf = df[df["timestamp"] <= ts]
            if len(sdf) == 0:
                continue
            prices[symbol] = float(sdf.iloc[-1]["close"])

        orders: list[dict[str, object]] = []
        positions = {p.symbol: p for p in self.broker.get_all_open_positions()}

        ranked_candidates = sorted(
            execution_symbols,
            key=lambda s: (target_weights.get(s, 0.0) - current_weights.get(s, 0.0), target_weights.get(s, 0.0)),
            reverse=True,
        )

        # 1) Sell first so capital is freed immediately.
        sell_candidates: list[tuple[str, float, float, float]] = []
        for symbol in execution_symbols:
            price = float(prices.get(symbol, np.nan))
            if not np.isfinite(price) or price <= 0:
                continue

            cw = float(current_weights.get(symbol, 0.0))
            tw = float(target_weights.get(symbol, 0.0))
            delta_w = tw - cw

            if symbol not in positions or abs(delta_w) < MIN_WEIGHT_DELTA:
                continue

            if tw <= FULL_EXIT_WEIGHT_THRESHOLD:
                desired_notional = cw * equity_before
                sell_notional = desired_notional
            elif delta_w < 0:
                urgency = float(np.clip(abs(delta_w) / 0.05, 0.75, 1.35))
                desired_notional = abs(delta_w) * equity_before
                sell_notional = desired_notional * MAX_SELL_FRACTION_PER_CYCLE * urgency
            else:
                continue

            if sell_notional >= MIN_NOTIONAL_TO_TRADE:
                sell_candidates.append((symbol, delta_w, sell_notional, price))

        sell_candidates.sort(key=lambda x: x[1])  # most negative delta first

        for symbol, delta_w, notional, price in sell_candidates:
            pos = positions.get(symbol)
            if pos is None or pos.qty <= 0 or self.broker.has_open_order_for_symbol(symbol):
                continue

            qty = min(int(pos.qty), int(notional // price))
            if qty <= 0 and abs(delta_w) > 0.03:
                qty = min(int(pos.qty), 1)

            if qty <= 0:
                continue

            self.broker.submit_market_sell(symbol, qty)
            self.broker.wait_until_qty_at_or_below(symbol, max(0.0, float(pos.qty) - qty))
            self._mark_order()

            orders.append(
                {
                    "trade_symbol": symbol,
                    "trade_side": "SELL",
                    "trade_qty": qty,
                    "trade_price": price,
                    "target_weight": tw,
                    "current_weight": cw,
                    "delta_weight": delta_w,
                    "action_value": float(action[stock_symbols.index(symbol)]) if symbol in stock_symbols else 0.0,
                    "message": "rebalance_sell",
                }
            )

        acct_mid = self._record_account_snapshot("after_sells", ts)
        equity_mid = float(acct_mid["equity"])
        buying_power_mid = float(acct_mid["buying_power"])

        # 2) Buy high-conviction names first.
        new_positions_count = 0
        buy_candidates = []
        for symbol in ranked_candidates:
            price = float(prices.get(symbol, np.nan))
            if not np.isfinite(price) or price <= 0:
                continue

            tw = float(target_weights.get(symbol, 0.0))
            cw = float(current_weights.get(symbol, 0.0))
            delta_w = tw - cw

            if delta_w < MIN_WEIGHT_DELTA:
                continue

            if symbol not in positions and tw <= 0.0:
                continue

            buy_notional = delta_w * equity_mid * buy_fraction * execution_scale
            if buy_notional >= MIN_NOTIONAL_TO_TRADE:
                buy_candidates.append((symbol, delta_w, tw, buy_notional, price))

        buy_candidates = buy_candidates[: max(TOP_CANDIDATES_FOR_NEW_BUYS, max_new_positions)]

        for symbol, delta_w, tw, notional, price in buy_candidates:
            if buying_power_mid < MIN_BUYING_POWER or self.broker.has_open_order_for_symbol(symbol):
                continue

            existing = self.broker.get_position(symbol)
            is_new = existing.qty <= 0
            if is_new and new_positions_count >= max_new_positions:
                continue

            max_affordable_notional = max(0.0, buying_power_mid - MIN_BUYING_POWER)
            if max_affordable_notional < MIN_NOTIONAL_TO_TRADE:
                continue

            qty = int(min(notional, max_affordable_notional) // price)
            if qty <= 0 and min(notional, max_affordable_notional) >= price:
                qty = 1

            if qty <= 0:
                continue

            self.broker.submit_market_buy(symbol, qty)
            self.broker.wait_for_position(symbol)
            self._mark_order()

            buying_power_mid = max(0.0, buying_power_mid - qty * price)
            if is_new:
                new_positions_count += 1

            orders.append(
                {
                    "trade_symbol": symbol,
                    "trade_side": "BUY",
                    "trade_qty": qty,
                    "trade_price": price,
                    "target_weight": tw,
                    "current_weight": current_weights.get(symbol, 0.0),
                    "delta_weight": delta_w,
                    "action_value": float(action[stock_symbols.index(symbol)]) if symbol in stock_symbols else 0.0,
                    "message": "rebalance_buy",
                }
            )

        acct_after = self._record_account_snapshot("after_execution", ts)
        for o in orders:
            o["equity_before"] = equity_before
            o["equity_after"] = float(acct_after["equity"])
            o["cash_before"] = cash_before
            o["cash_after"] = float(acct_after["cash"])
            o["buying_power_before"] = buying_power_before
            o["buying_power_after"] = float(acct_after["buying_power"])
            o["portfolio_value_model"] = ""
            o["signal_confidence"] = signal_confidence
            o["regime"] = regime
            o["regime_conf"] = regime_conf
            o["turbulence"] = ""
            o["turbulence_threshold"] = ""
            o["realized_volatility"] = realized_volatility
            o["execution_scale"] = execution_scale
            o.update(optimizer_diag)

        if not orders:
            orders.append(
                {
                    "message": "no_rebalance_needed",
                    "trade_symbol": "",
                    "trade_side": "",
                    "trade_qty": 0,
                    "trade_price": 0.0,
                    "target_weight": "",
                    "current_weight": "",
                    "delta_weight": "",
                    "action_value": "",
                    "equity_before": equity_before,
                    "equity_after": float(acct_after["equity"]),
                    "cash_before": cash_before,
                    "cash_after": float(acct_after["cash"]),
                    "buying_power_before": buying_power_before,
                    "buying_power_after": float(acct_after["buying_power"]),
                    "portfolio_value_model": "",
                    "signal_confidence": signal_confidence,
                    "regime": regime,
                    "regime_conf": regime_conf,
                    "turbulence": "",
                    "turbulence_threshold": "",
                    "realized_volatility": realized_volatility,
                    "execution_scale": execution_scale,
                    **optimizer_diag,
                }
            )

        return orders

    def _log_orders(self, ts: pd.Timestamp, row: pd.Series, model_info: dict[str, object], orders: list[dict[str, object]]) -> None:
        for order in orders:
            self.journal.append(
                {
                    "timestamp_utc": now_utc().isoformat(),
                    "bar_timestamp_et": "" if pd.isna(ts) else ts.isoformat(),
                    "equity_before": order.get("equity_before", ""),
                    "equity_after": order.get("equity_after", ""),
                    "cash_before": order.get("cash_before", ""),
                    "cash_after": order.get("cash_after", ""),
                    "buying_power_before": order.get("buying_power_before", ""),
                    "buying_power_after": order.get("buying_power_after", ""),
                    "portfolio_value_model": model_info.get("portfolio_value_model", ""),
                    "signal_confidence": order.get("signal_confidence", ""),
                    "regime": order.get("regime", ""),
                    "regime_conf": order.get("regime_conf", ""),
                    "turbulence": model_info.get("turbulence", ""),
                    "turbulence_threshold": model_info.get("turbulence_threshold", ""),
                    "realized_volatility": order.get("realized_volatility", ""),
                    "execution_scale": order.get("execution_scale", ""),
                    "optimizer_expected_return_annual": order.get("optimizer_expected_return_annual", ""),
                    "optimizer_expected_vol_annual": order.get("optimizer_expected_vol_annual", ""),
                    "optimizer_expected_sharpe": order.get("optimizer_expected_sharpe", ""),
                    "optimizer_weight_sum": order.get("optimizer_weight_sum", ""),
                    "optimizer_symbol_count": order.get("optimizer_symbol_count", ""),
                    "optimizer_portfolio_alpha": order.get("optimizer_portfolio_alpha", ""),
                    "optimizer_gross_exposure": order.get("optimizer_gross_exposure", ""),
                    "optimizer_per_asset_cap": order.get("optimizer_per_asset_cap", ""),
                    "optimizer_risk_aversion": order.get("optimizer_risk_aversion", ""),
                    "trade_symbol": order.get("trade_symbol", ""),
                    "trade_side": order.get("trade_side", ""),
                    "trade_qty": order.get("trade_qty", ""),
                    "trade_price": order.get("trade_price", ""),
                    "target_weight": order.get("target_weight", ""),
                    "current_weight": order.get("current_weight", ""),
                    "delta_weight": order.get("delta_weight", ""),
                    "action_value": order.get("action_value", ""),
                    "message": order.get("message", ""),
                }
            )

    def _print_heartbeat(
        self,
        latest_ts: pd.Timestamp,
        stock_symbols: list[str],
        portfolio_value_model: float,
        realized_volatility: float = 0.0,
        execution_scale: float = 1.0,
        optimizer_diag: Optional[dict[str, float]] = None,
    ) -> None:
        now_ts = time.time()
        if (now_ts - self.last_heartbeat_time) < HEARTBEAT_SECONDS:
            return

        optimizer_diag = optimizer_diag or {}
        equity_text = "n/a" if self.last_equity is None else f"${self.last_equity:,.2f}"
        print(
            f"[paper] heartbeat | latest_ts={latest_ts} | "
            f"tracked_symbols={len(stock_symbols)} | "
            f"broker_equity={equity_text} | "
            f"model_portfolio_value={portfolio_value_model:.2f} | "
            f"realized_vol={realized_volatility:.6f} | "
            f"exec_scale={execution_scale:.3f} | "
            f"opt_alpha={float(optimizer_diag.get('optimizer_portfolio_alpha', 0.0)):.4f} | "
            f"opt_sharpe={float(optimizer_diag.get('optimizer_expected_sharpe', 0.0)):.4f}"
        )
        self.last_heartbeat_time = now_ts

    def _process_completed_timestamp(self, ts: pd.Timestamp) -> None:
        if self.last_processed_timestamp is not None and ts <= self.last_processed_timestamp:
            return

        merged = self._build_merged_from_live_buffers(ts)
        if merged is None or len(merged) == 0:
            print("[paper] merged feature frame is empty before build_features")
            return

        feat = self.strategy.build_features(merged)
        self.strategy._load_vec_norm(feat)
        if len(feat) == 0:
            print("[paper] no feature rows available after live merge")
            return

        env = TradingEnv(data=feat.copy(), config=self.strategy.env_config)
        obs, _reset_info = env.reset()

        if self.strategy.vec_norm is not None:
            obs_in = self.strategy.vec_norm.normalize_obs(obs.reshape(1, -1))
        else:
            obs_in = obs.reshape(1, -1)

        action, _ = self.strategy.model.predict(obs_in, deterministic=True)
        action = np.asarray(action).reshape(-1)

        _obs2, reward, _done, _truncated, info = env.step(action)

        row = feat.iloc[env.idx]
        bar_ts = pd.to_datetime(row["timestamp"], errors="coerce")
        if pd.isna(bar_ts):
            print("[paper] invalid latest timestamp")
            return
        if not self._is_regular_hours_bar(bar_ts):
            self.last_processed_timestamp = ts
            print(f"[paper] skipping non-regular-hours bar: {bar_ts}")
            return

        stock_symbols = list(info["stock_symbols"])
        realized_volatility = self._estimate_market_volatility(bar_ts)
        execution_scale = self._volatility_execution_scale(
            realized_volatility=realized_volatility,
            regime=str(row.get("regime", "")),
            regime_conf=float(row.get("regime_conf", 0.0)),
            signal_confidence=float(row.get("signal_confidence", 0.0)),
        )

        optimizer_target_weights, _, optimizer_diag = self._optimizer_target_weights(
            ts=bar_ts,
            stock_symbols=stock_symbols,
            action=action,
            equity=float(self.broker.get_account_snapshot()["equity"]),
            regime=str(row.get("regime", "")),
            regime_conf=float(row.get("regime_conf", 0.0)),
            signal_confidence=float(row.get("signal_confidence", 0.0)),
        )

        top_targets = sorted(
            optimizer_target_weights.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:5]

        print(
            f"[model] ts={bar_ts} | regime={row.get('regime', 'UNKNOWN')} | "
            f"regime_conf={float(row.get('regime_conf', 0.0)):.3f} | "
            f"signal_conf={float(row.get('signal_confidence', 0.0)):.3f} | "
            f"portfolio_value={float(info['portfolio_value']):.2f} | "
            f"reward={float(reward):.5f} | "
            f"realized_vol={realized_volatility:.6f} | "
            f"exec_scale={execution_scale:.3f} | "
            f"opt_alpha={float(optimizer_diag.get('optimizer_portfolio_alpha', 0.0)):.4f} | "
            f"opt_sharpe={float(optimizer_diag.get('optimizer_expected_sharpe', 0.0)):.4f} | "
            f"top_targets={top_targets}"
        )

        orders = self._place_rebalance_orders(bar_ts, stock_symbols, action, row)

        model_info = {
            "portfolio_value_model": float(info["portfolio_value"]),
            "turbulence": float(info["turbulence"]),
            "turbulence_threshold": float(info["turbulence_threshold"]),
        }
        self._log_orders(bar_ts, row, model_info, orders)

        self.last_processed_timestamp = ts
        self._write_account_equity_curve()
        self._print_heartbeat(
            bar_ts,
            stock_symbols,
            float(info["portfolio_value"]),
            realized_volatility=realized_volatility,
            execution_scale=execution_scale,
            optimizer_diag=optimizer_diag,
        )

        for order in orders:
            print(
                f"[paper] {bar_ts} | {order.get('message','')} | "
                f"symbol={order.get('trade_symbol','')} | side={order.get('trade_side','')} | "
                f"qty={order.get('trade_qty','')} | price={order.get('trade_price','')} | "
                f"target_w={order.get('target_weight','')} | current_w={order.get('current_weight','')} | "
                f"delta_w={order.get('delta_weight','')} | "
                f"opt_alpha={order.get('optimizer_portfolio_alpha','')} | "
                f"opt_sharpe={order.get('optimizer_expected_sharpe','')} | "
                f"realized_vol={order.get('realized_volatility','')} | "
                f"exec_scale={order.get('execution_scale','')}"
            )

    def run(self) -> None:
        print("[paper] streaming-style paper trader starting (polling historical bars)...")
        print(f"[paper] journal={JOURNAL_PATH}")
        print(f"[paper] equity_curve={EQUITY_CURVE_PATH}")

        self._seed_live_buffers()
        self._record_account_snapshot("startup", None)
        self._write_account_equity_curve()

        while True:
            try:
                latest_ts = self._rebuild_from_poll()
                self._process_completed_timestamp(latest_ts)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[paper] polling error: {exc}")
            time.sleep(POLL_SECONDS)


def main() -> None:
    trader = ProductionPaperTrader()
    trader.run()


if __name__ == "__main__":
    main()
