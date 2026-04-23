from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AlphaPortfolioConfig:
    lookback_bars: int = 180
    min_history_rows: int = 60
    risk_free_rate_annual: float = 0.04
    cash_buffer_weight: float = 0.01

    # Alpha construction
    model_weight_power: float = 1.35
    recent_momentum_bars: int = 20
    medium_momentum_bars: int = 60
    alpha_model_weight: float = 0.65
    alpha_recent_momentum_weight: float = 0.25
    alpha_medium_momentum_weight: float = 0.10

    # Concentration and risk controls
    base_per_asset_weight_cap: float = 0.35
    max_per_asset_weight_cap: float = 0.60
    base_gross_exposure: float = 0.96
    max_gross_exposure: float = 0.99
    top_k: int = 5
    top_k_min_weight_share: float = 0.75
    shrinkage: float = 0.15
    turnover_penalty: float = 0.0005
    temperature: float = 0.45
    max_iter: int = 3000
    step_size: float = 0.08


class AlphaPortfolioOptimizer:
    """
    Alpha-first long-only allocator.

    Design goals:
      - trust the model to identify opportunities
      - allocate across both held names and new candidates
      - concentrate capital into the highest-alpha names
      - use covariance only as a risk-aware regularizer, not as the alpha source
    """

    def __init__(self, config: Optional[AlphaPortfolioConfig] = None):
        self.config = config or AlphaPortfolioConfig()

    @staticmethod
    def _annualization_factor_from_bars_per_day(
        bars_per_day: float = 390.0,
        trading_days: float = 252.0,
    ) -> float:
        return float(bars_per_day * trading_days)

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x
        mu = float(np.nanmean(x))
        sigma = float(np.nanstd(x))
        if not np.isfinite(sigma) or sigma < 1e-12:
            return np.zeros_like(x, dtype=float)
        return (x - mu) / sigma

    @staticmethod
    def _project_to_simplex_with_caps(
        w: np.ndarray,
        total: float,
        cap: float,
        floor: float = 0.0,
        max_iter: int = 500,
    ) -> np.ndarray:
        w = np.asarray(w, dtype=float).copy()
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        n = len(w)
        cap = float(max(cap, floor))
        total = float(np.clip(total, 0.0, n * cap))

        w = np.clip(w, floor, cap)

        for _ in range(max_iter):
            s = float(w.sum())
            diff = total - s
            if abs(diff) < 1e-10:
                break

            if diff > 0:
                free = w < (cap - 1e-12)
            else:
                free = w > (floor + 1e-12)

            free_count = int(free.sum())
            if free_count <= 0:
                break

            w[free] += diff / free_count
            w = np.clip(w, floor, cap)

        s = float(w.sum())
        if s > 1e-12:
            w *= total / s
            w = np.clip(w, floor, cap)

        return w

    def _build_close_matrix(
        self,
        price_history: pd.DataFrame,
        symbols: list[str],
    ) -> pd.DataFrame:
        required = {"timestamp", "symbol", "close"}
        missing = required - set(price_history.columns)
        if missing:
            raise ValueError(f"price_history missing required columns: {sorted(missing)}")

        df = price_history.copy()
        df = df[df["symbol"].isin(symbols)].copy()
        if df.empty:
            raise ValueError("No rows remain after filtering price history by symbols")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "symbol", "close"])
        df = df.sort_values(["timestamp", "symbol"])

        closes = (
            df.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
            .sort_index()
            .ffill()
            .dropna(axis=1, how="all")
        )
        return closes.tail(self.config.lookback_bars)

    def _estimate_returns_and_cov(
        self,
        closes: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        if closes.shape[0] < self.config.min_history_rows:
            raise ValueError("Not enough history for optimization")

        rets = closes.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
        rets = rets.dropna(axis=1, how="any")

        if rets.shape[0] < 20 or rets.shape[1] == 0:
            raise ValueError("Return matrix is empty after cleaning")

        ann_factor = self._annualization_factor_from_bars_per_day()
        sample_cov = rets.cov().values * ann_factor
        diag_cov = np.diag(np.diag(sample_cov))
        shrink = float(np.clip(self.config.shrinkage, 0.0, 1.0))
        cov = (1.0 - shrink) * sample_cov + shrink * diag_cov

        return rets, cov, list(rets.columns)

    def _dynamic_gross_exposure(self, regime: str, regime_conf: float, signal_confidence: float) -> float:
        regime = str(regime or "").upper()
        regime_conf = float(np.clip(regime_conf, 0.0, 1.0))
        signal_confidence = float(np.clip(signal_confidence, 0.0, 1.0))

        if regime == "BULL":
            bonus = 0.02 + 0.02 * regime_conf + 0.01 * signal_confidence
        elif regime == "TRANSITION":
            bonus = -0.05 + 0.02 * signal_confidence
        else:
            bonus = -0.12 + 0.04 * signal_confidence

        gross = self.config.base_gross_exposure + bonus
        return float(np.clip(gross, 0.70, self.config.max_gross_exposure))

    def _dynamic_cap(self, regime: str, regime_conf: float, signal_confidence: float) -> float:
        regime = str(regime or "").upper()
        regime_conf = float(np.clip(regime_conf, 0.0, 1.0))
        signal_confidence = float(np.clip(signal_confidence, 0.0, 1.0))

        cap = self.config.base_per_asset_weight_cap
        if regime == "BULL":
            cap += 0.10 * regime_conf + 0.08 * signal_confidence
        elif regime == "TRANSITION":
            cap += 0.02 * signal_confidence
        else:
            cap -= 0.10 * (1.0 - signal_confidence)

        return float(np.clip(cap, 0.12, self.config.max_per_asset_weight_cap))

    def _build_alpha_scores(
        self,
        rets: pd.DataFrame,
        symbols: list[str],
        model_target_weights: dict[str, float],
    ) -> np.ndarray:
        model_prior = np.asarray([float(model_target_weights.get(s, 0.0)) for s in symbols], dtype=float)
        model_prior = np.clip(model_prior, 0.0, None)
        if model_prior.sum() > 1e-12:
            model_component = model_prior / model_prior.sum()
        else:
            model_component = np.zeros_like(model_prior)

        model_component = np.power(model_component, self.config.model_weight_power)
        model_component = self._zscore(model_component)

        recent_n = max(2, min(self.config.recent_momentum_bars, len(rets)))
        medium_n = max(2, min(self.config.medium_momentum_bars, len(rets)))

        recent_mom = (1.0 + rets.tail(recent_n)).prod(axis=0).values - 1.0
        medium_mom = (1.0 + rets.tail(medium_n)).prod(axis=0).values - 1.0

        recent_component = self._zscore(recent_mom)
        medium_component = self._zscore(medium_mom)

        alpha = (
            self.config.alpha_model_weight * model_component
            + self.config.alpha_recent_momentum_weight * recent_component
            + self.config.alpha_medium_momentum_weight * medium_component
        )

        # Make sure names with zero model support are still disadvantaged.
        alpha += 0.25 * np.sign(model_prior)
        return np.asarray(alpha, dtype=float)

    def _initial_weights(
        self,
        symbols: list[str],
        alpha: np.ndarray,
        current_weights: dict[str, float],
        gross_exposure: float,
        per_asset_cap: float,
    ) -> np.ndarray:
        current_w = np.asarray([float(current_weights.get(s, 0.0)) for s in symbols], dtype=float)
        current_w = np.clip(current_w, 0.0, None)

        top_k = max(1, min(self.config.top_k, len(symbols)))
        order = np.argsort(alpha)[::-1]
        mask = np.zeros(len(symbols), dtype=float)
        mask[order[:top_k]] = 1.0

        scaled_alpha = alpha / max(self.config.temperature, 1e-6)
        scaled_alpha = scaled_alpha - float(np.nanmax(scaled_alpha))
        soft = np.exp(np.clip(scaled_alpha, -50.0, 50.0))
        soft = soft * (1.0 + 1.25 * mask)

        if soft.sum() <= 1e-12:
            soft = np.ones_like(soft)

        target = gross_exposure * soft / soft.sum()

        # Give some inertia to currently held positions, but keep it light.
        if current_w.sum() > 1e-12:
            current_w = current_w / current_w.sum() * gross_exposure
            target = 0.85 * target + 0.15 * current_w

        target = self._project_to_simplex_with_caps(
            target,
            total=gross_exposure,
            cap=per_asset_cap,
            floor=0.0,
        )

        # Push the top-k concentration share upward.
        top_share = float(target[order[:top_k]].sum())
        desired_top_share = float(np.clip(self.config.top_k_min_weight_share, 0.0, 1.0)) * gross_exposure
        if top_share < desired_top_share:
            extra_needed = desired_top_share - top_share
            target[order[:top_k]] += extra_needed / top_k
            target = self._project_to_simplex_with_caps(
                target,
                total=gross_exposure,
                cap=per_asset_cap,
                floor=0.0,
            )

        return target

    def _objective(
        self,
        w: np.ndarray,
        alpha: np.ndarray,
        cov: np.ndarray,
        current_w: np.ndarray,
        risk_aversion: float,
    ) -> float:
        alpha_term = float(w @ alpha)
        risk_term = float(w @ cov @ w)
        turnover = float(np.abs(w - current_w).sum())
        penalty = self.config.turnover_penalty * turnover
        return alpha_term - risk_aversion * risk_term - penalty

    def _optimize_weights(
        self,
        alpha: np.ndarray,
        cov: np.ndarray,
        current_w: np.ndarray,
        initial_w: np.ndarray,
        gross_exposure: float,
        per_asset_cap: float,
        risk_aversion: float,
    ) -> np.ndarray:
        w = np.asarray(initial_w, dtype=float).copy()
        best_w = w.copy()
        best_score = self._objective(best_w, alpha, cov, current_w, risk_aversion)
        step = float(self.config.step_size)

        for _ in range(self.config.max_iter):
            grad = alpha - 2.0 * risk_aversion * (cov @ w)
            grad -= self.config.turnover_penalty * np.sign(w - current_w)

            candidate = w + step * grad
            candidate = self._project_to_simplex_with_caps(
                candidate,
                total=gross_exposure,
                cap=per_asset_cap,
                floor=0.0,
            )

            score = self._objective(candidate, alpha, cov, current_w, risk_aversion)
            if score > best_score:
                best_score = score
                best_w = candidate.copy()
                w = candidate
                step = min(step * 1.01, 0.25)
            else:
                step *= 0.995

            if step < 1e-5:
                break

        return best_w

    def optimize(
        self,
        price_history: pd.DataFrame,
        tradable_symbols: list[str],
        current_weights: dict[str, float],
        model_target_weights: Optional[dict[str, float]] = None,
        regime: str = "",
        regime_conf: float = 0.0,
        signal_confidence: float = 0.0,
    ) -> tuple[dict[str, float], dict[str, float]]:
        model_target_weights = model_target_weights or {}

        closes = self._build_close_matrix(price_history=price_history, symbols=tradable_symbols)
        rets, cov, symbols = self._estimate_returns_and_cov(closes)
        alpha = self._build_alpha_scores(rets, symbols, model_target_weights)

        gross_exposure = self._dynamic_gross_exposure(regime, regime_conf, signal_confidence)
        gross_exposure = min(gross_exposure, max(0.0, 1.0 - self.config.cash_buffer_weight))
        per_asset_cap = self._dynamic_cap(regime, regime_conf, signal_confidence)

        current_w = np.asarray([float(current_weights.get(s, 0.0)) for s in symbols], dtype=float)
        initial_w = self._initial_weights(
            symbols=symbols,
            alpha=alpha,
            current_weights=current_weights,
            gross_exposure=gross_exposure,
            per_asset_cap=per_asset_cap,
        )

        # Lower risk aversion in strong bullish/high-confidence conditions.
        regime_upper = str(regime or "").upper()
        risk_aversion = 0.10
        if regime_upper == "BULL":
            risk_aversion = max(0.03, 0.10 - 0.05 * float(np.clip(regime_conf, 0.0, 1.0)) - 0.03 * float(np.clip(signal_confidence, 0.0, 1.0)))
        elif regime_upper == "TRANSITION":
            risk_aversion = 0.12
        else:
            risk_aversion = 0.18

        final_w = self._optimize_weights(
            alpha=alpha,
            cov=cov,
            current_w=current_w,
            initial_w=initial_w,
            gross_exposure=gross_exposure,
            per_asset_cap=per_asset_cap,
            risk_aversion=risk_aversion,
        )

        port_alpha = float(final_w @ alpha)
        port_vol = float(np.sqrt(max(final_w @ cov @ final_w, 1e-12)))
        ann_factor = self._annualization_factor_from_bars_per_day()
        mean_ret = rets.mean().reindex(symbols).values * ann_factor
        exp_ret = float(final_w @ mean_ret)
        exp_sharpe = (exp_ret - self.config.risk_free_rate_annual) / max(port_vol, 1e-12)

        alpha_by_symbol = {s: float(a) for s, a in zip(symbols, alpha)}
        weights = {s: float(w) for s, w in zip(symbols, final_w)}

        diag = {
            "optimizer_expected_return_annual": exp_ret,
            "optimizer_expected_vol_annual": port_vol,
            "optimizer_expected_sharpe": exp_sharpe,
            "optimizer_weight_sum": float(final_w.sum()),
            "optimizer_symbol_count": float(len(symbols)),
            "optimizer_portfolio_alpha": port_alpha,
            "optimizer_gross_exposure": gross_exposure,
            "optimizer_per_asset_cap": per_asset_cap,
            "optimizer_risk_aversion": risk_aversion,
            "optimizer_top_weight": float(final_w.max()) if len(final_w) else 0.0,
        }
        diag.update({f"alpha_{k}": v for k, v in alpha_by_symbol.items()})
        return weights, diag
