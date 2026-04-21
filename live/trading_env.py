from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class TradeCentricMDPConfig:
    initial_cash: float = 100_000.0
    hmax: int = 100
    transaction_cost_pct: float = 0.001
    invalid_action_penalty: float = 0.001
    turbulence_threshold_quantile: float = 0.99
    max_episode_steps: int = 256
    allow_fractional_clip_to_cash: bool = True
    reward_scale: float = 1.0
    min_feature_lookback: int = 30
    target_num_stocks: int = 30

    # Trade-centric shaping
    hold_winner_bonus_weight: float = 0.20
    strong_trend_adx_threshold: float = 20.0
    strong_trend_macd_floor: float = 0.0

    loser_hold_penalty_weight: float = 0.15
    loser_hold_threshold: float = 0.005

    stagnation_penalty_weight: float = 0.002
    stagnation_threshold: float = 0.002
    stagnation_bars_threshold: int = 8

    small_exit_penalty: float = 0.005
    small_exit_threshold: float = 0.0075

    premature_exit_penalty_weight: float = 0.75
    premature_exit_lookahead: int = 5
    premature_exit_min_future_gain: float = 0.0075

    trade_reward_weight: float = 1.00
    positive_trade_exponent: float = 1.25
    negative_trade_linear_weight: float = 1.25

    mfe_capture_bonus_weight: float = 0.90
    min_bars_for_capture_bonus: int = 3
    no_position_cash_idle_penalty: float = 0.0


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: pd.DataFrame, config: Optional[TradeCentricMDPConfig] = None):
        super().__init__()
        self.config = config or TradeCentricMDPConfig()
        self.data = data.reset_index(drop=True).copy()

        self._discover_stock_universe()
        self._build_indicator_matrices()
        self._build_turbulence_index()

        if self.num_stocks == 0:
            raise ValueError("No tradable stock close columns were found in the input dataframe.")

        self.extra_context_cols = [
            c for c in self.data.columns
            if c.startswith("universe_")
            or c in {"regime_conf", "bull_score", "bear_score", "transition_score", "signal_confidence"}
        ]

        obs_dim = 1 + (6 * self.num_stocks) + len(self.extra_context_cols)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_stocks,),
            dtype=np.float32,
        )

        self.episode_start_idx = 0
        self.idx = 0
        self.episode_step = 0

        self.balance = float(self.config.initial_cash)
        self.holdings = np.zeros(self.num_stocks, dtype=np.int64)
        self.prev_portfolio_value = float(self.config.initial_cash)

        self.avg_entry_price = np.zeros(self.num_stocks, dtype=np.float64)
        self.trade_bars_held = np.zeros(self.num_stocks, dtype=np.int64)
        self.trade_mfe = np.zeros(self.num_stocks, dtype=np.float64)
        self.trade_active = np.zeros(self.num_stocks, dtype=bool)

    def _discover_stock_universe(self) -> None:
        close_cols = [c for c in self.data.columns if c.endswith("_close") and c not in {"close"}]
        preferred_exclude = {"qqq_close"}
        candidate_close_cols = [c for c in close_cols if c not in preferred_exclude]
        candidate_close_cols = sorted(candidate_close_cols)

        if self.config.target_num_stocks > 0:
            candidate_close_cols = candidate_close_cols[: self.config.target_num_stocks]

        self.stock_close_cols = candidate_close_cols
        self.stock_symbols = [c[:-6].upper() for c in self.stock_close_cols]
        self.stock_open_cols = [f"{s.lower()}_open" for s in self.stock_symbols]
        self.stock_high_cols = [f"{s.lower()}_high" for s in self.stock_symbols]
        self.stock_low_cols = [f"{s.lower()}_low" for s in self.stock_symbols]
        self.stock_volume_cols = [f"{s.lower()}_volume" for s in self.stock_symbols]
        self.num_stocks = len(self.stock_symbols)

        missing_core = []
        for col in self.stock_open_cols + self.stock_high_cols + self.stock_low_cols + self.stock_volume_cols:
            if col not in self.data.columns:
                missing_core.append(col)
        if missing_core:
            raise ValueError(
                f"Input dataframe is missing OHLCV columns needed for multi-stock indicators: {missing_core[:20]}"
            )

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _compute_macd(self, close: pd.Series) -> pd.Series:
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd_line = ema12 - ema26
        signal = self._ema(macd_line, 9)
        return (macd_line - signal).replace([np.inf, -np.inf], np.nan)

    def _compute_rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        return (100.0 - (100.0 / (1.0 + rs))).replace([np.inf, -np.inf], np.nan)

    def _compute_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        tp = (high + low + close) / 3.0
        sma_tp = tp.rolling(window).mean()
        mad = (tp - sma_tp).abs().rolling(window).mean()
        return ((tp - sma_tp) / (0.015 * (mad + 1e-8))).replace([np.inf, -np.inf], np.nan)

    def _compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(window).mean()
        plus_di = 100.0 * plus_dm.rolling(window).mean() / (atr + 1e-8)
        minus_di = 100.0 * minus_dm.rolling(window).mean() / (atr + 1e-8)
        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8))
        return dx.rolling(window).mean().replace([np.inf, -np.inf], np.nan)

    def _build_indicator_matrices(self) -> None:
        prices, macd, rsi, cci, adx = [], [], [], [], []

        for sym in self.stock_symbols:
            close = self.data[f"{sym.lower()}_close"].astype(float)
            high = self.data[f"{sym.lower()}_high"].astype(float)
            low = self.data[f"{sym.lower()}_low"].astype(float)

            prices.append(close.rename(sym))
            macd.append(self._compute_macd(close).rename(sym))
            rsi.append(self._compute_rsi(close).rename(sym))
            cci.append(self._compute_cci(high, low, close).rename(sym))
            adx.append(self._compute_adx(high, low, close).rename(sym))

        self.price_matrix = pd.concat(prices, axis=1).ffill().bfill()
        self.macd_matrix = pd.concat(macd, axis=1).fillna(0.0)
        self.rsi_matrix = pd.concat(rsi, axis=1).fillna(50.0)
        self.cci_matrix = pd.concat(cci, axis=1).fillna(0.0)
        self.adx_matrix = pd.concat(adx, axis=1).fillna(0.0)

        start_idx = max(self.config.min_feature_lookback, 30)
        self.valid_start_idx = min(start_idx, max(len(self.data) - 2, 0))

    def _build_turbulence_index(self) -> None:
        returns = self.price_matrix.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        turbulence = np.zeros(len(self.data), dtype=float)
        for i in range(len(self.data)):
            if i < 60:
                continue

            hist = returns.iloc[max(0, i - 252):i]
            if len(hist) < 20:
                continue

            cov = hist.cov().to_numpy()
            try:
                cov_inv = np.linalg.pinv(cov)
            except Exception:
                continue

            y = returns.iloc[i].to_numpy()
            mu = hist.mean().to_numpy()
            diff = y - mu
            turbulence[i] = float(diff.T @ cov_inv @ diff)

        self.turbulence = pd.Series(turbulence, index=self.data.index)
        self.turbulence_threshold = float(self.turbulence.quantile(self.config.turbulence_threshold_quantile))

    def _row(self) -> pd.Series:
        return self.data.iloc[self.idx]

    def _prices_t(self) -> np.ndarray:
        return self.price_matrix.iloc[self.idx].to_numpy(dtype=np.float32)

    def _prices_tp1(self) -> np.ndarray:
        j = min(self.idx + 1, len(self.price_matrix) - 1)
        return self.price_matrix.iloc[j].to_numpy(dtype=np.float32)

    def _macd_t(self) -> np.ndarray:
        return self.macd_matrix.iloc[self.idx].to_numpy(dtype=np.float32)

    def _rsi_t(self) -> np.ndarray:
        return self.rsi_matrix.iloc[self.idx].to_numpy(dtype=np.float32)

    def _cci_t(self) -> np.ndarray:
        return self.cci_matrix.iloc[self.idx].to_numpy(dtype=np.float32)

    def _adx_t(self) -> np.ndarray:
        return self.adx_matrix.iloc[self.idx].to_numpy(dtype=np.float32)

    def _portfolio_value(self, prices: np.ndarray) -> float:
        return float(self.balance + np.dot(prices, self.holdings))

    def _in_turbulence(self) -> bool:
        return float(self.turbulence.iloc[self.idx]) > self.turbulence_threshold

    def _get_obs(self) -> np.ndarray:
        extras = [float(self._row()[c]) for c in self.extra_context_cols]
        obs = np.concatenate(
            [
                np.array([self.balance], dtype=np.float32),
                self._prices_t(),
                self.holdings.astype(np.float32),
                self._macd_t(),
                self._rsi_t(),
                self._cci_t(),
                self._adx_t(),
                np.asarray(extras, dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    def _future_continuation_regret(self, stock_idx: int, exit_price: float) -> float:
        horizon = self.config.premature_exit_lookahead
        end_idx = min(self.idx + horizon, len(self.price_matrix) - 1)
        if end_idx <= self.idx:
            return 0.0

        future_prices = self.price_matrix.iloc[self.idx + 1:end_idx + 1, stock_idx].to_numpy(dtype=np.float64)
        if future_prices.size == 0:
            return 0.0

        best_future = float(np.max(future_prices))
        future_gain = (best_future - exit_price) / max(exit_price, 1e-8)
        return max(0.0, future_gain)

    def _trade_terminal_bonus(self, realized_return: float) -> float:
        if realized_return > 0:
            return self.config.trade_reward_weight * (realized_return ** self.config.positive_trade_exponent)
        return -self.config.trade_reward_weight * (
            abs(realized_return) * self.config.negative_trade_linear_weight
        )

    def _capture_bonus(self, realized_return: float, mfe: float, bars_held: int) -> float:
        if bars_held < self.config.min_bars_for_capture_bonus or mfe <= 1e-8:
            return 0.0
        capture = np.clip(realized_return / mfe, 0.0, 1.0)
        return self.config.mfe_capture_bonus_weight * capture

    def _trade_exit_shaping(self, stock_idx: int, exit_price: float) -> float:
        if not self.trade_active[stock_idx] or self.avg_entry_price[stock_idx] <= 0:
            return 0.0

        realized_return = (exit_price - self.avg_entry_price[stock_idx]) / max(self.avg_entry_price[stock_idx], 1e-8)
        bars_held = int(self.trade_bars_held[stock_idx])
        mfe = float(self.trade_mfe[stock_idx])

        bonus = self._trade_terminal_bonus(realized_return)
        bonus += self._capture_bonus(realized_return, mfe, bars_held)

        if 0.0 < realized_return < self.config.small_exit_threshold:
            bonus -= self.config.small_exit_penalty

        regret = self._future_continuation_regret(stock_idx, exit_price)
        if regret >= self.config.premature_exit_min_future_gain:
            bonus -= self.config.premature_exit_penalty_weight * regret

        return float(bonus)

    def _clear_trade_tracking(self, stock_idx: int) -> None:
        self.avg_entry_price[stock_idx] = 0.0
        self.trade_bars_held[stock_idx] = 0
        self.trade_mfe[stock_idx] = 0.0
        self.trade_active[stock_idx] = False

    def _mark_trade_state(self, prices_t: np.ndarray) -> None:
        active_idx = np.where(self.holdings > 0)[0]
        for i in active_idx:
            if not self.trade_active[i]:
                self.trade_active[i] = True
                self.avg_entry_price[i] = float(prices_t[i])
                self.trade_bars_held[i] = 0
                self.trade_mfe[i] = 0.0
            else:
                self.trade_bars_held[i] += 1
                rtn = (float(prices_t[i]) - self.avg_entry_price[i]) / max(self.avg_entry_price[i], 1e-8)
                self.trade_mfe[i] = max(self.trade_mfe[i], rtn)

    def _sell_all_due_to_turbulence(self, prices_t: np.ndarray) -> float:
        if np.sum(self.holdings) <= 0:
            return 0.0

        reward_adjustment = 0.0
        for i in range(self.num_stocks):
            if self.holdings[i] > 0:
                reward_adjustment += self._trade_exit_shaping(i, float(prices_t[i]))

        sell_values = prices_t * self.holdings.astype(np.float32)
        gross = float(np.sum(sell_values))
        cost = gross * self.config.transaction_cost_pct

        self.balance += gross - cost
        self.holdings[:] = 0
        reward_adjustment -= cost

        for i in range(self.num_stocks):
            self._clear_trade_tracking(i)

        return reward_adjustment

    def _apply_continuous_action(self, action: np.ndarray, prices_t: np.ndarray) -> float:
        action = np.clip(action.astype(np.float32), -1.0, 1.0)

        trade_qty = np.rint(np.abs(action) * self.config.hmax).astype(np.int64)
        buy_mask = action > 0
        sell_mask = action < 0

        reward_adjustment = 0.0

        if np.any(sell_mask):
            desired_sell_qty = trade_qty * sell_mask.astype(np.int64)
            sell_qty = np.minimum(desired_sell_qty, self.holdings)

            for i in np.where(sell_qty > 0)[0]:
                if sell_qty[i] >= self.holdings[i]:
                    reward_adjustment += self._trade_exit_shaping(i, float(prices_t[i]))

            sell_values = prices_t * sell_qty.astype(np.float32)
            gross_sell = float(np.sum(sell_values))
            sell_cost = gross_sell * self.config.transaction_cost_pct

            self.holdings -= sell_qty
            self.balance += gross_sell - sell_cost
            reward_adjustment -= sell_cost

            for i in np.where(self.holdings <= 0)[0]:
                if self.trade_active[i]:
                    self._clear_trade_tracking(i)

        if np.any(buy_mask):
            desired_buy_qty = trade_qty * buy_mask.astype(np.int64)
            desired_buy_values = prices_t * desired_buy_qty.astype(np.float32)
            desired_gross = float(np.sum(desired_buy_values))
            desired_cost = desired_gross * self.config.transaction_cost_pct
            desired_total = desired_gross + desired_cost

            if desired_total <= self.balance + 1e-8:
                buy_qty = desired_buy_qty
            else:
                if self.config.allow_fractional_clip_to_cash and desired_total > 0:
                    scale = self.balance / desired_total
                    buy_qty = np.floor(desired_buy_qty.astype(np.float32) * scale).astype(np.int64)
                else:
                    buy_qty = np.zeros_like(desired_buy_qty)

            buy_values = prices_t * buy_qty.astype(np.float32)
            gross_buy = float(np.sum(buy_values))
            buy_cost = gross_buy * self.config.transaction_cost_pct
            total_buy = gross_buy + buy_cost

            if total_buy > self.balance + 1e-8:
                reward_adjustment -= self.config.invalid_action_penalty
            else:
                new_idx = np.where((buy_qty > 0) & (self.holdings == 0))[0]
                self.holdings += buy_qty
                self.balance -= total_buy
                reward_adjustment -= buy_cost
                for i in new_idx:
                    self.trade_active[i] = True
                    self.avg_entry_price[i] = float(prices_t[i])
                    self.trade_bars_held[i] = 0
                    self.trade_mfe[i] = 0.0

        if self.balance < -1e-6:
            reward_adjustment -= self.config.invalid_action_penalty
            self.balance = max(self.balance, 0.0)

        return reward_adjustment

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        upper = max(len(self.data) - self.config.max_episode_steps - 1, self.valid_start_idx + 1)
        self.episode_start_idx = int(self.np_random.integers(self.valid_start_idx, upper))
        self.idx = self.episode_start_idx
        self.episode_step = 0

        self.balance = float(self.config.initial_cash)
        self.holdings = np.zeros(self.num_stocks, dtype=np.int64)
        self.prev_portfolio_value = float(self.config.initial_cash)

        self.avg_entry_price = np.zeros(self.num_stocks, dtype=np.float64)
        self.trade_bars_held = np.zeros(self.num_stocks, dtype=np.int64)
        self.trade_mfe = np.zeros(self.num_stocks, dtype=np.float64)
        self.trade_active = np.zeros(self.num_stocks, dtype=bool)

        return self._get_obs(), {
            "stock_symbols": self.stock_symbols,
            "turbulence_threshold": self.turbulence_threshold,
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        done = False
        truncated = False
        reward = 0.0
        info: dict[str, object] = {}

        prices_t = self._prices_t()
        self._mark_trade_state(prices_t)
        value_t = self._portfolio_value(prices_t)

        if self._in_turbulence():
            reward += self._sell_all_due_to_turbulence(prices_t)
            action = np.zeros_like(action, dtype=np.float32)
        else:
            reward += self._apply_continuous_action(action, prices_t)

        active_idx = np.where(self.trade_active)[0]
        for i in active_idx:
            rtn = (float(prices_t[i]) - self.avg_entry_price[i]) / max(self.avg_entry_price[i], 1e-8)
            macd_ok = float(self.macd_matrix.iloc[self.idx, i]) >= self.config.strong_trend_macd_floor
            adx_ok = float(self.adx_matrix.iloc[self.idx, i]) >= self.config.strong_trend_adx_threshold

            if rtn > 0 and macd_ok and adx_ok:
                reward += self.config.hold_winner_bonus_weight * min(rtn, 0.05)

            if rtn < -self.config.loser_hold_threshold:
                reward -= self.config.loser_hold_penalty_weight * (abs(rtn) - self.config.loser_hold_threshold)

            if (
                self.trade_bars_held[i] >= self.config.stagnation_bars_threshold
                and abs(rtn) < self.config.stagnation_threshold
            ):
                reward -= self.config.stagnation_penalty_weight

        self.idx += 1
        self.episode_step += 1

        if self.idx >= len(self.data) - 1:
            done = True
            self.idx = min(self.idx, len(self.data) - 1)

        prices_tp1 = self._prices_tp1()
        value_tp1 = self._portfolio_value(prices_tp1)

        step_portfolio_change = value_tp1 - value_t
        reward += step_portfolio_change * self.config.reward_scale

        if np.sum(self.holdings) == 0 and self.config.no_position_cash_idle_penalty > 0:
            reward -= self.config.no_position_cash_idle_penalty

        if self.episode_step >= self.config.max_episode_steps:
            truncated = True

        self.prev_portfolio_value = value_tp1

        info["portfolio_value"] = float(value_tp1)
        info["balance"] = float(self.balance)
        info["holdings"] = self.holdings.copy()
        info["prices"] = prices_tp1.copy()
        info["step_portfolio_change"] = float(step_portfolio_change)
        info["turbulence"] = float(self.turbulence.iloc[self.idx])
        info["turbulence_threshold"] = float(self.turbulence_threshold)
        info["stock_symbols"] = self.stock_symbols

        return self._get_obs(), float(reward), done, truncated, info
