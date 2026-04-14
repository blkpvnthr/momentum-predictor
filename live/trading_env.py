from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Regime-directed trading environment.

    Intended behavior:
    - In BULL regimes, the agent may only enter long TQQQ.
    - In BEAR regimes, the agent may only enter long SQQQ.
    - QQQ is the signal anchor only. Its price action / derived features drive
      entries, but QQQ itself is never traded.
    - TQQQ and SQQQ prices are used for execution and PnL accounting.

    Actions:
        0 = HOLD
        1 = ENTER_TQQQ
        2 = EXIT_TQQQ
        3 = ENTER_SQQQ
        4 = EXIT_SQQQ
    """

    metadata = {"render_modes": []}

    SYMBOL_FLAT = 0
    SYMBOL_TQQQ = 1
    SYMBOL_SQQQ = 2

    def __init__(
        self,
        data: pd.DataFrame,
        min_hold_bars: int = 10,
        transaction_cost: float = 0.0020,
        slippage_cost: float = 0.0015,
        churn_penalty: float = 0.0040,
        invalid_action_penalty: float = 0.01,
        regime_violation_penalty: float = 0.05,
        low_confidence_entry_penalty: float = 0.01,
        min_signal_confidence: float = 0.60,
        flat_reward: float = 0.0002,
        holding_penalty: float = 0.00005,
        transition_holding_penalty: float = 0.0025,
        tqqq_extra_penalty: float = 0.0010,
        sma_gate_penalty: float = 0.01,
        tqqq_bull_strength_threshold: float = 0.75,
        sqqq_bear_strength_threshold: float = 0.75,
        adx_threshold: float = 18.0,
        max_episode_steps: int = 120,
        rolling_sharpe_window: int = 20,
        sharpe_weight: float = 0.10,
        downside_penalty_weight: float = 0.15,
        drawdown_penalty_weight: float = 0.15,
    ):
        super().__init__()

        self.data = data.reset_index(drop=True).copy()

        self.min_hold_bars = int(min_hold_bars)
        self.transaction_cost = float(transaction_cost)
        self.slippage_cost = float(slippage_cost)
        self.churn_penalty = float(churn_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.regime_violation_penalty = float(regime_violation_penalty)
        self.low_confidence_entry_penalty = float(low_confidence_entry_penalty)
        self.min_signal_confidence = float(min_signal_confidence)
        self.flat_reward = float(flat_reward)
        self.holding_penalty = float(holding_penalty)
        self.transition_holding_penalty = float(transition_holding_penalty)
        self.tqqq_extra_penalty = float(tqqq_extra_penalty)
        self.sma_gate_penalty = float(sma_gate_penalty)
        self.tqqq_bull_strength_threshold = float(tqqq_bull_strength_threshold)
        self.sqqq_bear_strength_threshold = float(sqqq_bear_strength_threshold)
        self.adx_threshold = float(adx_threshold)

        self.max_episode_steps = int(max_episode_steps)
        self.rolling_sharpe_window = int(rolling_sharpe_window)
        self.sharpe_weight = float(sharpe_weight)
        self.downside_penalty_weight = float(downside_penalty_weight)
        self.drawdown_penalty_weight = float(drawdown_penalty_weight)

        self.obs_cols = [
            "pred_ret_5",
            "pred_ret_15",
            "pred_ret_30",
            "breakout_prob",
            "continuation_prob",
            "signal_confidence",
            "momentum_score",
            "momentum_dispersion",
            "momentum_agreement",
            "price_vs_sma20",
            "tema20_slope",
            "adx",
            "atr_pct",
            "atr_expansion",
            "mfi_14",
            "bop",
            "obv_slope",
            "donchian_breakout",
            "donchian_breakdown",
            "donchian_width",
            "donchian_distance_up",
            "donchian_distance_down",
            "regime_conf",
            "bull_score",
            "bear_score",
            "transition_score",
            "is_opening_window",
            "is_midday",
            "is_power_hour",
            "hour",
            "minute",
            "sma_20",
            "sma_50",
            "sma20_slope",
            "bull_cross_state",
            "bear_cross_state",
            "cross_up_event",
            "cross_down_event",
        ]

        required_cols = self.obs_cols + [
            "regime",
            "qqq_close",
            "tqqq_close",
            "sqqq_close",
        ]
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"Environment data missing required columns: {missing}")

        self.action_space = spaces.Discrete(5)
        obs_dim = len(self.obs_cols) + 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.episode_start_idx = 0
        self.idx = 0
        self.episode_step = 0

        self.position_symbol = self.SYMBOL_FLAT
        self.entry_price = 0.0
        self.prev_mark_to_market = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_trade = 0
        self.max_favorable_excursion = 0.0
        self.current_drawdown = 0.0

        self.returns_window: deque[float] = deque(maxlen=self.rolling_sharpe_window)

    def _row(self) -> pd.Series:
        return self.data.iloc[self.idx]

    def _signal_confidence(self) -> float:
        return float(self._row()["signal_confidence"])

    def _is_flat(self) -> bool:
        return self.position_symbol == self.SYMBOL_FLAT

    def _is_tqqq(self) -> bool:
        return self.position_symbol == self.SYMBOL_TQQQ

    def _is_sqqq(self) -> bool:
        return self.position_symbol == self.SYMBOL_SQQQ

    def _current_symbol_close(self) -> float:
        row = self._row()

        if self._is_tqqq():
            return float(row["tqqq_close"])
        if self._is_sqqq():
            return float(row["sqqq_close"])
        return float(row["qqq_close"])

    def _entry_price_for_symbol(self, symbol_code: int) -> float:
        row = self._row()

        if symbol_code == self.SYMBOL_TQQQ:
            return float(row["tqqq_close"])
        if symbol_code == self.SYMBOL_SQQQ:
            return float(row["sqqq_close"])

        raise ValueError(f"Unsupported symbol code: {symbol_code}")

    def _mark_to_market(self, price: float) -> float:
        if self._is_flat():
            return 0.0
        return (price - self.entry_price) / max(self.entry_price, 1e-8)

    def _bull_sma_ok(self) -> bool:
        row = self._row()
        return bool(
            (float(row["sma_20"]) > float(row["sma_50"]))
            and (float(row["bull_cross_state"]) > 0.5)
            and (float(row["price_vs_sma20"]) > 0.0)
            and (float(row["sma20_slope"]) > 0.0)
        )

    def _bear_sma_ok(self) -> bool:
        row = self._row()
        return bool(
            (float(row["sma_20"]) < float(row["sma_50"]))
            and (float(row["bear_cross_state"]) > 0.5)
            and (float(row["price_vs_sma20"]) < 0.0)
            and (float(row["sma20_slope"]) < 0.0)
        )

    def _adx_ok(self) -> bool:
        adx = float(self._row()["adx"])
        return (not np.isnan(adx)) and (adx >= self.adx_threshold)

    def _tqqq_allowed(self) -> bool:
        row = self._row()
        return bool(
            str(row["regime"]) == "BULL"
            and self._bull_sma_ok()
            and float(row["bull_score"]) >= self.tqqq_bull_strength_threshold
            and float(row["signal_confidence"]) >= self.min_signal_confidence
            and self._adx_ok()
        )

    def _sqqq_allowed(self) -> bool:
        row = self._row()
        return bool(
            str(row["regime"]) == "BEAR"
            and self._bear_sma_ok()
            and float(row["bear_score"]) >= self.sqqq_bear_strength_threshold
            and float(row["signal_confidence"]) >= self.min_signal_confidence
            and self._adx_ok()
        )

    def _allowed_actions(self) -> np.ndarray:
        mask = np.zeros(5, dtype=bool)
        mask[0] = True  # HOLD

        if self._is_flat():
            if self._tqqq_allowed():
                mask[1] = True  # ENTER_TQQQ
            if self._sqqq_allowed():
                mask[3] = True  # ENTER_SQQQ

        elif self._is_tqqq():
            mask[2] = True  # EXIT_TQQQ

        elif self._is_sqqq():
            mask[4] = True  # EXIT_SQQQ

        return mask

    def _get_obs(self) -> np.ndarray:
        row = self._row()

        base_features = [float(row[col]) for col in self.obs_cols]
        controlled_state = [
            float(self.position_symbol == self.SYMBOL_TQQQ),
            float(self.position_symbol == self.SYMBOL_SQQQ),
            float(self.unrealized_pnl),
            float(self.steps_in_trade),
            float(self.current_drawdown),
        ]

        return np.asarray(base_features + controlled_state, dtype=np.float32)

    def _reset_trade_state(self) -> None:
        self.position_symbol = self.SYMBOL_FLAT
        self.entry_price = 0.0
        self.prev_mark_to_market = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_trade = 0
        self.max_favorable_excursion = 0.0
        self.current_drawdown = 0.0

    def _enter_position(self, symbol_code: int) -> None:
        self.position_symbol = symbol_code
        self.entry_price = self._entry_price_for_symbol(symbol_code)
        self.prev_mark_to_market = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_trade = 0
        self.max_favorable_excursion = 0.0
        self.current_drawdown = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        event_rows = self.data.index[
            (self.data["donchian_breakout"] == 1.0)
            | (self.data["donchian_breakdown"] == 1.0)
            | (self.data["cross_up_event"] == 1.0)
            | (self.data["cross_down_event"] == 1.0)
        ].tolist()

        if event_rows:
            self.episode_start_idx = int(self.np_random.choice(event_rows))
        else:
            upper = max(len(self.data) - self.max_episode_steps - 1, 1)
            self.episode_start_idx = int(self.np_random.integers(0, upper))

        self.idx = self.episode_start_idx
        self.episode_step = 0
        self.returns_window.clear()
        self._reset_trade_state()

        return self._get_obs(), {"action_mask": self._allowed_actions()}

    def step(self, action: int):
        action = int(action)

        info: dict[str, object] = {}
        done = False
        truncated = False
        reward = 0.0

        current_row = self._row()
        current_price = self._current_symbol_close()
        regime = str(current_row["regime"])
        signal_confidence = self._signal_confidence()

        allowed = self._allowed_actions()
        valid_action = bool(allowed[action])

        if not valid_action:
            reward -= self.invalid_action_penalty
            action = 0

        mtm_before = self._mark_to_market(current_price)

        # Entry / exit logic
        if action == 1 and self._is_flat():  # ENTER_TQQQ
            if regime != "BULL":
                reward -= self.regime_violation_penalty
            if signal_confidence < self.min_signal_confidence:
                reward -= self.low_confidence_entry_penalty
            if not self._tqqq_allowed():
                reward -= self.sma_gate_penalty

            self._enter_position(self.SYMBOL_TQQQ)
            reward -= self.transaction_cost + self.slippage_cost + self.tqqq_extra_penalty

        elif action == 2 and self._is_tqqq():  # EXIT_TQQQ
            realized = mtm_before
            reward += realized
            reward -= self.transaction_cost + self.slippage_cost
            if self.steps_in_trade < self.min_hold_bars:
                reward -= self.churn_penalty
            self._reset_trade_state()

        elif action == 3 and self._is_flat():  # ENTER_SQQQ
            if regime != "BEAR":
                reward -= self.regime_violation_penalty
            if signal_confidence < self.min_signal_confidence:
                reward -= self.low_confidence_entry_penalty
            if not self._sqqq_allowed():
                reward -= self.sma_gate_penalty

            self._enter_position(self.SYMBOL_SQQQ)
            reward -= self.transaction_cost + self.slippage_cost

        elif action == 4 and self._is_sqqq():  # EXIT_SQQQ
            realized = mtm_before
            reward += realized
            reward -= self.transaction_cost + self.slippage_cost
            if self.steps_in_trade < self.min_hold_bars:
                reward -= self.churn_penalty
            self._reset_trade_state()

        # Advance time
        self.idx += 1
        self.episode_step += 1

        if self.idx >= len(self.data) - 1:
            done = True
            self.idx = min(self.idx, len(self.data) - 1)

        next_row = self._row()
        next_price = self._current_symbol_close()

        mtm_after = self._mark_to_market(next_price)
        pnl_delta = mtm_after - self.prev_mark_to_market

        # Reward shaping
        if not self._is_flat():
            self.steps_in_trade += 1
            self.unrealized_pnl = mtm_after
            self.max_favorable_excursion = max(self.max_favorable_excursion, mtm_after)
            self.current_drawdown = self.max_favorable_excursion - mtm_after

            reward += pnl_delta
            reward -= self.holding_penalty

            self.returns_window.append(pnl_delta)
            returns_arr = np.asarray(self.returns_window, dtype=np.float32)

            if len(returns_arr) >= 5:
                mean_r = float(np.mean(returns_arr))
                std_r = float(np.std(returns_arr)) + 1e-8
                rolling_sharpe = mean_r / std_r

                downside = returns_arr[returns_arr < 0]
                downside_std = float(np.std(downside)) if len(downside) > 0 else 0.0

                reward += self.sharpe_weight * rolling_sharpe
                reward -= self.downside_penalty_weight * downside_std
                reward -= self.drawdown_penalty_weight * self.current_drawdown
        else:
            self.unrealized_pnl = 0.0
            self.steps_in_trade = 0
            self.max_favorable_excursion = 0.0
            self.current_drawdown = 0.0
            self.returns_window.append(0.0)
            reward += self.flat_reward

        self.prev_mark_to_market = mtm_after

        if not self._is_flat() and str(next_row["regime"]) == "TRANSITION":
            reward -= self.transition_holding_penalty

        if self.episode_step >= self.max_episode_steps:
            truncated = True

        info["action_mask"] = self._allowed_actions()
        info["position_symbol"] = self.position_symbol
        info["unrealized_pnl"] = self.unrealized_pnl
        info["regime"] = str(next_row["regime"])
        info["signal_confidence"] = float(next_row["signal_confidence"])

        return self._get_obs(), float(reward), done, truncated, info