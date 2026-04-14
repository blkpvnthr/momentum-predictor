from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Regime-directed trading environment.

    Action map:
        0 = HOLD
        1 = ENTER_TQQQ_10
        2 = ENTER_TQQQ_15
        3 = UNUSED
        4 = EXIT_TQQQ
        5 = ENTER_SQQQ_10
        6 = ENTER_SQQQ_15
        7 = UNUSED
        8 = EXIT_SQQQ
    """

    metadata = {"render_modes": []}

    SYMBOL_FLAT = 0
    SYMBOL_TQQQ = 1
    SYMBOL_SQQQ = 2

    TQQQ_ENTRY_ACTIONS = {1: 0.10, 2: 0.15}
    SQQQ_ENTRY_ACTIONS = {5: 0.10, 6: 0.15}

    def __init__(
        self,
        data: pd.DataFrame,
        min_hold_bars: int = 20,
        transaction_cost: float = 0.0015,
        slippage_cost: float = 0.0010,
        invalid_action_penalty: float = 0.01,
        low_confidence_entry_penalty: float = 0.004,
        weak_edge_penalty: float = 0.006,
        forced_exit_penalty: float = 0.050,
        churn_penalty: float = 0.004,
        min_signal_confidence: float = 0.78,
        no_trade_signal_confidence: float = 0.75,
        no_trade_regime_confidence: float = 0.70,
        no_trade_edge_threshold: float = 0.25,
        flat_reward: float = 0.0,
        winner_hold_bonus: float = 0.20,
        loser_hold_penalty: float = 0.35,
        holding_penalty: float = 0.00008,
        transition_holding_penalty: float = 0.0040,
        tqqq_extra_penalty: float = 0.0010,
        sma_gate_penalty: float = 0.004,
        tqqq_bull_strength_threshold: float = 0.78,
        sqqq_bear_strength_threshold: float = 0.72,
        adx_threshold: float = 20.0,
        max_episode_steps: int = 120,
        cooldown_bars: int = 30,
        drawdown_penalty_weight: float = 0.12,
        reward_vol_normalization: bool = True,
        score_edge_threshold: float = 0.30,
    ):
        super().__init__()

        self.data = data.reset_index(drop=True).copy()

        self.min_hold_bars = int(min_hold_bars)
        self.transaction_cost = float(transaction_cost)
        self.slippage_cost = float(slippage_cost)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.low_confidence_entry_penalty = float(low_confidence_entry_penalty)
        self.weak_edge_penalty = float(weak_edge_penalty)
        self.forced_exit_penalty = float(forced_exit_penalty)
        self.churn_penalty = float(churn_penalty)
        self.min_signal_confidence = float(min_signal_confidence)
        self.no_trade_signal_confidence = float(no_trade_signal_confidence)
        self.no_trade_regime_confidence = float(no_trade_regime_confidence)
        self.no_trade_edge_threshold = float(no_trade_edge_threshold)
        self.flat_reward = float(flat_reward)
        self.winner_hold_bonus = float(winner_hold_bonus)
        self.loser_hold_penalty = float(loser_hold_penalty)
        self.holding_penalty = float(holding_penalty)
        self.transition_holding_penalty = float(transition_holding_penalty)
        self.tqqq_extra_penalty = float(tqqq_extra_penalty)
        self.sma_gate_penalty = float(sma_gate_penalty)
        self.tqqq_bull_strength_threshold = float(tqqq_bull_strength_threshold)
        self.sqqq_bear_strength_threshold = float(sqqq_bear_strength_threshold)
        self.adx_threshold = float(adx_threshold)
        self.max_episode_steps = int(max_episode_steps)
        self.cooldown_bars = int(cooldown_bars)
        self.drawdown_penalty_weight = float(drawdown_penalty_weight)
        self.reward_vol_normalization = bool(reward_vol_normalization)
        self.score_edge_threshold = float(score_edge_threshold)

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

        required_cols = self.obs_cols + ["regime", "qqq_close", "tqqq_close", "sqqq_close"]
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"Environment data missing required columns: {missing}")

        self.action_space = spaces.Discrete(9)
        obs_dim = len(self.obs_cols) + 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.episode_start_idx = 0
        self.idx = 0
        self.episode_step = 0
        self.last_trade_step = -10_000

        self.position_symbol = self.SYMBOL_FLAT
        self.position_size = 0.0
        self.entry_price = 0.0
        self.prev_mark_to_market = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_trade = 0
        self.max_favorable_excursion = 0.0
        self.current_drawdown = 0.0

    def _row(self) -> pd.Series:
        return self.data.iloc[self.idx]

    def _signal_confidence(self) -> float:
        return float(self._row()["signal_confidence"])

    def _is_flat(self) -> bool:
        return self.position_symbol == self.SYMBOL_FLAT or self.position_size <= 0.0

    def _is_tqqq(self) -> bool:
        return self.position_symbol == self.SYMBOL_TQQQ and self.position_size > 0.0

    def _is_sqqq(self) -> bool:
        return self.position_symbol == self.SYMBOL_SQQQ and self.position_size > 0.0

    def _in_cooldown(self) -> bool:
        return (self.episode_step - self.last_trade_step) < self.cooldown_bars

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
        raw_ret = (price - self.entry_price) / max(self.entry_price, 1e-8)
        return self.position_size * raw_ret

    def _bull_signal_score(self) -> float:
        row = self._row()
        score = 0.0
        score += 0.35 * float(row["bull_score"])
        score += 0.20 * float(row["signal_confidence"])
        score += 0.15 * float(float(row["price_vs_sma20"]) > 0.0)
        score += 0.10 * float(float(row["sma20_slope"]) > 0.0)
        score += 0.10 * float(float(row["bull_cross_state"]) > 0.5)
        score += 0.10 * float(float(row["momentum_score"]) > 0.0)
        return float(score)

    def _bear_signal_score(self) -> float:
        row = self._row()
        score = 0.0
        score += 0.35 * float(row["bear_score"])
        score += 0.20 * float(row["signal_confidence"])
        score += 0.15 * float(float(row["price_vs_sma20"]) < 0.0)
        score += 0.10 * float(float(row["sma20_slope"]) < 0.0)
        score += 0.10 * float(float(row["bear_cross_state"]) > 0.5)
        score += 0.10 * float(float(row["momentum_score"]) < 0.0)
        return float(score)

    def _score_edge(self) -> float:
        return self._bull_signal_score() - self._bear_signal_score()

    def _adx_ok(self) -> bool:
        adx = float(self._row()["adx"])
        return (not np.isnan(adx)) and (adx >= self.adx_threshold)

    def _no_trade_zone(self) -> bool:
        row = self._row()
        return bool(
            abs(self._score_edge()) < self.no_trade_edge_threshold
            or float(row["signal_confidence"]) < self.no_trade_signal_confidence
            or float(row["regime_conf"]) < self.no_trade_regime_confidence
        )

    def _bull_signal_ok(self) -> bool:
        row = self._row()
        edge = self._score_edge()
        return bool(
            float(row["bull_score"]) >= self.tqqq_bull_strength_threshold
            and float(row["signal_confidence"]) >= self.min_signal_confidence
            and float(row["price_vs_sma20"]) > 0.0
            and float(row["sma20_slope"]) > 0.0
            and float(row["bull_cross_state"]) > 0.5
            and float(row["momentum_score"]) > 0.05
            and self._adx_ok()
            and edge >= self.score_edge_threshold
        )

    def _bear_signal_ok(self) -> bool:
        row = self._row()
        edge = self._score_edge()
        return bool(
            float(row["bear_score"]) >= self.sqqq_bear_strength_threshold
            and float(row["signal_confidence"]) >= 0.70
            and float(row["price_vs_sma20"]) < 0.0
            and float(row["sma20_slope"]) < 0.0
            and float(row["bear_cross_state"]) > 0.5
            and float(row["momentum_score"]) < 0.0
            and self._adx_ok()
            and edge <= -0.20
        )

    def _allowed_actions(self) -> np.ndarray:
        mask = np.zeros(9, dtype=bool)
        mask[0] = True

        if self._in_cooldown() or self._no_trade_zone():
            return mask

        regime = str(self._row()["regime"])

        if self._is_flat():
            if regime == "BULL":
                mask[1] = True
                mask[2] = True
            if regime == "BEAR":
                mask[5] = True
                mask[6] = True
        elif self._is_tqqq():
            mask[4] = True
        elif self._is_sqqq():
            mask[8] = True

        return mask

    def _get_obs(self) -> np.ndarray:
        row = self._row()
        base_features = [float(row[col]) for col in self.obs_cols]
        controlled_state = [
            float(self.position_symbol == self.SYMBOL_TQQQ),
            float(self.position_symbol == self.SYMBOL_SQQQ),
            float(self.position_size),
            float(self.unrealized_pnl),
            float(self.steps_in_trade),
            float(self.current_drawdown),
        ]
        return np.asarray(base_features + controlled_state, dtype=np.float32)

    def _reset_trade_state(self) -> None:
        self.position_symbol = self.SYMBOL_FLAT
        self.position_size = 0.0
        self.entry_price = 0.0
        self.prev_mark_to_market = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_trade = 0
        self.max_favorable_excursion = 0.0
        self.current_drawdown = 0.0

    def _enter_position(self, symbol_code: int, size: float) -> None:
        self.position_symbol = symbol_code
        self.position_size = float(np.clip(size, 0.0, 1.0))
        self.entry_price = self._entry_price_for_symbol(symbol_code)
        self.prev_mark_to_market = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_trade = 0
        self.max_favorable_excursion = 0.0
        self.current_drawdown = 0.0
        self.last_trade_step = self.episode_step

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
        self.last_trade_step = -10_000
        self._reset_trade_state()

        return self._get_obs(), {"action_mask": self._allowed_actions()}

    def step(self, action: int):
        action = int(action)

        info: dict[str, object] = {}
        done = False
        truncated = False

        current_row = self._row()
        current_price = self._current_symbol_close()
        regime = str(current_row["regime"])
        signal_confidence = self._signal_confidence()
        edge_now = self._score_edge()

        # 35% sizing actions are unused
        if action in (3, 7):
            action = 0

        allowed = self._allowed_actions()
        valid_action = bool(allowed[action])

        mtm_before = self._mark_to_market(current_price)
        reward = 0.0
        trade_executed = False
        clean_exit = False
        forced_exit = False

        if not valid_action:
            reward -= self.invalid_action_penalty
            action = 0

        # =========================================================
        # ENTRY LOGIC
        # =========================================================
        if self._is_flat():
            if self._no_trade_zone():
                action = 0

            if action in self.TQQQ_ENTRY_ACTIONS:
                requested_size = self.TQQQ_ENTRY_ACTIONS[action]

                if (
                    float(current_row["bear_score"]) >= float(current_row["bull_score"]) - 0.08
                    or float(current_row["momentum_score"]) <= 0.0
                    or float(current_row["sma20_slope"]) <= 0.0
                    or float(current_row["price_vs_sma20"]) <= 0.0
                    or edge_now < self.score_edge_threshold
                ):
                    action = 0
                elif regime == "BULL" and self._bull_signal_ok():
                    self._enter_position(self.SYMBOL_TQQQ, requested_size)
                    trade_executed = True
                    reward -= (
                        self.transaction_cost
                        + self.slippage_cost
                        + self.tqqq_extra_penalty
                        + self.churn_penalty
                    )

                    if signal_confidence < self.min_signal_confidence:
                        reward -= self.low_confidence_entry_penalty
                    if abs(edge_now) < self.score_edge_threshold:
                        reward -= self.weak_edge_penalty
                    if not self._bull_signal_ok():
                        reward -= self.sma_gate_penalty

            elif action in self.SQQQ_ENTRY_ACTIONS:
                requested_size = self.SQQQ_ENTRY_ACTIONS[action]

                if (
                    float(current_row["bull_score"]) >= float(current_row["bear_score"]) - 0.03
                    or float(current_row["price_vs_sma20"]) >= 0.0
                    or float(current_row["sma20_slope"]) >= 0.0
                    or edge_now > -0.20
                ):
                    action = 0
                elif regime == "BEAR" and self._bear_signal_ok():
                    self._enter_position(self.SYMBOL_SQQQ, requested_size)
                    trade_executed = True
                    reward -= self.transaction_cost + self.slippage_cost + self.churn_penalty

                    if signal_confidence < 0.70:
                        reward -= self.low_confidence_entry_penalty
                    if abs(edge_now) < 0.20:
                        reward -= self.weak_edge_penalty
                    if not self._bear_signal_ok():
                        reward -= self.sma_gate_penalty

        # =========================================================
        # VOLUNTARY EXIT LOGIC
        # =========================================================
        elif action == 4 and self._is_tqqq():
            if self.steps_in_trade < self.min_hold_bars:
                action = 0
            else:
                realized = mtm_before
                trade_executed = True
                clean_exit = True
                reward += realized
                reward -= self.transaction_cost + self.slippage_cost + self.churn_penalty

                exit_quality = (
                    abs(edge_now) < 0.20
                    or regime != "BULL"
                    or float(current_row["bull_score"]) < 0.70
                )
                if exit_quality:
                    reward += 0.05

                self._reset_trade_state()
                self.last_trade_step = self.episode_step

        elif action == 8 and self._is_sqqq():
            if self.steps_in_trade < self.min_hold_bars:
                action = 0
            else:
                realized = mtm_before
                trade_executed = True
                clean_exit = True
                reward += realized
                reward -= self.transaction_cost + self.slippage_cost + self.churn_penalty

                exit_quality = (
                    abs(edge_now) < 0.20
                    or regime != "BEAR"
                    or float(current_row["bear_score"]) < 0.70
                )
                if exit_quality:
                    reward += 0.05

                self._reset_trade_state()
                self.last_trade_step = self.episode_step

        # =========================================================
        # ADVANCE TIME
        # =========================================================
        self.idx += 1
        self.episode_step += 1

        if self.idx >= len(self.data) - 1:
            done = True
            self.idx = min(self.idx, len(self.data) - 1)

        next_row = self._row()
        next_price = self._current_symbol_close()

        # =========================================================
        # FORCED REGIME EXITS
        # =========================================================
        if self._is_tqqq() and str(next_row["regime"]) == "BEAR":
            mtm_forced = self._mark_to_market(next_price)
            forced_exit = True
            reward += mtm_forced
            reward -= self.forced_exit_penalty
            self._reset_trade_state()
            self.last_trade_step = self.episode_step

        elif self._is_sqqq() and str(next_row["regime"]) == "BULL":
            mtm_forced = self._mark_to_market(next_price)
            forced_exit = True
            reward += mtm_forced
            reward -= self.forced_exit_penalty
            self._reset_trade_state()
            self.last_trade_step = self.episode_step

        # =========================================================
        # HOLDING REWARD / PENALTY
        # =========================================================
        mtm_after = self._mark_to_market(next_price)
        pnl_delta = mtm_after - self.prev_mark_to_market

        if not self._is_flat():
            self.steps_in_trade += 1
            self.unrealized_pnl = mtm_after
            self.max_favorable_excursion = max(self.max_favorable_excursion, mtm_after)
            self.current_drawdown = self.max_favorable_excursion - mtm_after

            reward_core = pnl_delta
            if self.reward_vol_normalization:
                vol = float(next_row.get("atr_pct", 0.0))
                reward_core = reward_core / max(vol, 1e-6)

            reward += reward_core
            reward -= self.holding_penalty
            reward -= self.drawdown_penalty_weight * self.current_drawdown

            if mtm_after > 0:
                reward += self.winner_hold_bonus * reward_core
            else:
                reward += self.loser_hold_penalty * reward_core

            if str(next_row["regime"]) == "TRANSITION":
                reward -= self.transition_holding_penalty

            # Stale-trade penalty: discourage sitting forever until forced exit
            if self.steps_in_trade > 12:
                reward -= 0.0015 * float(self.steps_in_trade - 12)

        else:
            self.unrealized_pnl = 0.0
            self.steps_in_trade = 0
            self.max_favorable_excursion = 0.0
            self.current_drawdown = 0.0
            reward += self.flat_reward

        self.prev_mark_to_market = mtm_after

        if self.episode_step >= self.max_episode_steps:
            truncated = True

        info["action_mask"] = self._allowed_actions()
        info["position_symbol"] = self.position_symbol
        info["position_size"] = self.position_size
        info["unrealized_pnl"] = self.unrealized_pnl
        info["regime"] = str(next_row["regime"])
        info["signal_confidence"] = float(next_row["signal_confidence"])
        info["trade_executed"] = trade_executed
        info["clean_exit"] = clean_exit
        info["forced_exit"] = forced_exit

        return self._get_obs(), float(reward), done, truncated, info