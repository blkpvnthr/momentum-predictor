from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trading_env import TradeCentricMDPConfig, TradingEnv

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover
    GaussianHMM = None


THIS_FILE = Path(__file__).resolve()
LIVE_DIR = THIS_FILE.parent
PROJECT_ROOT = LIVE_DIR.parent

MODEL_PATH = LIVE_DIR / "trading_model.zip"
VECNORM_PATH = LIVE_DIR / "vec_normalize.pkl"
REGIME_ENGINE_PATH = LIVE_DIR / "regime_engine.pkl"


@dataclass
class HMMRegimeConfig:
    n_states: int = 3
    covariance_type: str = "diag"
    n_iter: int = 200
    tol: float = 1e-2
    random_state: int = 42
    bull_state_name: str = "BULL"
    bear_state_name: str = "BEAR"
    transition_state_name: str = "TRANSITION"


class HMMRegimeEngine:
    def __init__(self, config: HMMRegimeConfig | None = None):
        self.config = config or HMMRegimeConfig()
        self.model: GaussianHMM | None = None
        self.scaler: StandardScaler | None = None
        self.feature_cols: list[str] = []
        self.state_to_regime: dict[int, str] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "HMMRegimeEngine":
        if GaussianHMM is None:
            raise RuntimeError("hmmlearn is not available. Install it to use the HMM regime engine.")

        self.feature_cols = list(feature_cols)
        x = (
            df[self.feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(x) < 200:
            raise RuntimeError("Not enough rows to fit HMM regime engine.")

        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x)

        self.model = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            random_state=self.config.random_state,
        )
        self.model.fit(x_scaled)

        hidden = self.model.predict(x_scaled)
        temp = pd.DataFrame(x, columns=self.feature_cols)
        temp["state"] = hidden

        ret_col = "ret_1" if "ret_1" in temp.columns else ("ret_5" if "ret_5" in temp.columns else self.feature_cols[0])
        agg_map: dict[str, tuple[str, str]] = {
            "mean_ret": (ret_col, "mean"),
            "vol": ("atr_pct", "mean") if "atr_pct" in temp.columns else (ret_col, "std"),
            "adx_mean": ("adx", "mean") if "adx" in temp.columns else (ret_col, "mean"),
        }
        state_summary = temp.groupby("state").agg(**agg_map).reset_index()

        bull_state = int(state_summary.sort_values("mean_ret", ascending=False).iloc[0]["state"])
        bear_state = int(state_summary.sort_values("mean_ret", ascending=True).iloc[0]["state"])

        self.state_to_regime = {}
        for state in state_summary["state"].tolist():
            s = int(state)
            if s == bull_state:
                self.state_to_regime[s] = self.config.bull_state_name
            elif s == bear_state:
                self.state_to_regime[s] = self.config.bear_state_name
            else:
                self.state_to_regime[s] = self.config.transition_state_name

        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("HMMRegimeEngine must be fitted before predict().")

        temp = df.copy()
        x = (
            temp[self.feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        x_scaled = self.scaler.transform(x) if self.scaler is not None else x
        hidden = self.model.predict(x_scaled)
        probs = self.model.predict_proba(x_scaled)

        temp["hmm_state"] = hidden
        temp["regime"] = [self.state_to_regime.get(int(s), "TRANSITION") for s in hidden]
        temp["regime_conf"] = probs.max(axis=1).astype(np.float32)
        temp["bull_score"] = 0.0
        temp["bear_score"] = 0.0
        temp["transition_score"] = 0.0

        for i, state in enumerate(hidden):
            regime = self.state_to_regime.get(int(state), "TRANSITION")
            confidence = float(probs[i].max())
            idx = temp.index[i]
            if regime == "BULL":
                temp.at[idx, "bull_score"] = confidence
            elif regime == "BEAR":
                temp.at[idx, "bear_score"] = confidence
            else:
                temp.at[idx, "transition_score"] = confidence

        return temp

    def save(self, path: str | Path) -> None:
        payload = {
            "config": self.config,
            "feature_cols": self.feature_cols,
            "state_to_regime": self.state_to_regime,
            "model": self.model,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
        }
        with Path(path).open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "HMMRegimeEngine":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)

        obj = cls(payload["config"])
        obj.feature_cols = payload["feature_cols"]
        obj.state_to_regime = payload["state_to_regime"]
        obj.model = payload["model"]
        obj.scaler = payload.get("scaler")
        obj.is_fitted = payload["is_fitted"]
        return obj


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-8) -> pd.Series:
    return a / (b + eps)


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return _safe_div(series - mean, std).replace([np.inf, -np.inf], np.nan)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _compute_adx(df: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(df)
    atr = tr.rolling(window).mean()

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(window).mean() / (atr + 1e-8)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(window).mean() / (atr + 1e-8)

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8))
    adx = dx.rolling(window).mean()
    return adx, plus_di, minus_di


def _build_universe_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close_cols = [c for c in df.columns if c.endswith("_close") and c not in {"close"}]
    volume_cols = [c for c in df.columns if c.endswith("_volume") and c not in {"volume"}]

    if not close_cols:
        return df

    close_df = df[close_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    volume_df = df[volume_cols].astype(float).replace([np.inf, -np.inf], np.nan) if volume_cols else pd.DataFrame(index=df.index)

    ret_1 = close_df.pct_change(1)
    ret_5 = close_df.pct_change(5)
    ret_15 = close_df.pct_change(15)

    sma20 = close_df.rolling(20, min_periods=5).mean()
    above_sma20 = (close_df > sma20).astype(float)

    vol_ma20 = volume_df.rolling(20, min_periods=5).mean() if not volume_df.empty else pd.DataFrame(index=df.index)
    vol_ratio = (volume_df / vol_ma20.replace(0.0, np.nan)) if not volume_df.empty else pd.DataFrame(index=df.index)

    top_k = max(1, int(np.ceil(close_df.shape[1] * 0.10)))

    universe = pd.DataFrame(index=df.index)
    universe["universe_ret_mean_1m"] = ret_1.mean(axis=1).fillna(0.0)
    universe["universe_ret_mean_5m"] = ret_5.mean(axis=1).fillna(0.0)
    universe["universe_ret_mean_15m"] = ret_15.mean(axis=1).fillna(0.0)

    universe["universe_breadth_up_1m"] = (ret_1 > 0).mean(axis=1).fillna(0.5)
    universe["universe_breadth_up_5m"] = (ret_5 > 0).mean(axis=1).fillna(0.5)
    universe["universe_breadth_up_15m"] = (ret_15 > 0).mean(axis=1).fillna(0.5)

    universe["universe_breadth_down_5m"] = (ret_5 < 0).mean(axis=1).fillna(0.5)
    universe["universe_trend_breadth"] = above_sma20.mean(axis=1).fillna(0.5)

    universe["universe_dispersion_1m"] = ret_1.std(axis=1).fillna(0.0)
    universe["universe_dispersion_5m"] = ret_5.std(axis=1).fillna(0.0)
    universe["universe_dispersion_15m"] = ret_15.std(axis=1).fillna(0.0)

    def _row_top_mean(row: pd.Series) -> float:
        vals = np.sort(row.dropna().values)
        return 0.0 if len(vals) == 0 else float(np.nanmean(vals[-top_k:]))

    def _row_bottom_mean(row: pd.Series) -> float:
        vals = np.sort(row.dropna().values)
        return 0.0 if len(vals) == 0 else float(np.nanmean(vals[:top_k]))

    universe["universe_leadership_score"] = ret_5.apply(_row_top_mean, axis=1).fillna(0.0)
    universe["universe_laggard_score"] = ret_5.apply(_row_bottom_mean, axis=1).fillna(0.0)
    universe["universe_leader_minus_laggard"] = (
        universe["universe_leadership_score"] - universe["universe_laggard_score"]
    ).fillna(0.0)

    if not vol_ratio.empty:
        universe["universe_volume_pressure"] = (
            vol_ratio.clip(lower=0.0, upper=5.0).mean(axis=1) - 1.0
        ).fillna(0.0)
        universe["universe_volume_breadth"] = (vol_ratio > 1.1).mean(axis=1).fillna(0.0)
    else:
        universe["universe_volume_pressure"] = 0.0
        universe["universe_volume_breadth"] = 0.0

    universe = universe.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return pd.concat([df, universe], axis=1)


def build_features(
    df: pd.DataFrame,
    regime_engine: HMMRegimeEngine | None = None,
    fit_regime_engine: bool = False,
) -> tuple[pd.DataFrame, HMMRegimeEngine | None]:
    df = df.copy()

    required = ["timestamp", "open", "high", "low", "close", "volume", "tqqq_close", "sqqq_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["qqq_open"] = df["open"]
    df["qqq_high"] = df["high"]
    df["qqq_low"] = df["low"]
    df["qqq_close"] = df["close"]
    df["qqq_volume"] = df["volume"]

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_15"] = df["close"].pct_change(15)
    df["ret_30"] = df["close"].pct_change(30)
    df["log_return"] = np.log(df["close"]).diff()

    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    ema_20 = _ema(df["close"], 20)
    df["tema_20"] = 3 * ema_20 - 3 * _ema(ema_20, 20) + _ema(_ema(ema_20, 20), 20)

    df["price_vs_sma20"] = _safe_div(df["close"] - df["sma_20"], df["sma_20"])
    df["tema20_slope"] = df["tema_20"].pct_change(3)
    df["sma20_slope"] = df["sma_20"].pct_change(3)

    df["bull_cross_state"] = (df["sma_20"] > df["sma_50"]).astype(float)
    df["bear_cross_state"] = (df["sma_20"] < df["sma_50"]).astype(float)
    df["cross_up_event"] = (
        (df["sma_20"] > df["sma_50"])
        & (df["sma_20"].shift(1) <= df["sma_50"].shift(1))
    ).astype(float)
    df["cross_down_event"] = (
        (df["sma_20"] < df["sma_50"])
        & (df["sma_20"].shift(1) >= df["sma_50"].shift(1))
    ).astype(float)

    df["roc_10"] = df["close"].pct_change(10) * 100.0

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100.0 * _safe_div(df["close"] - low_14, high_14 - low_14)

    ema_12 = _ema(df["close"], 12)
    ema_26 = _ema(df["close"], 26)
    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = _ema(df["macd_line"], 9)
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = tp.rolling(20).mean()
    mad = (tp - sma_tp).abs().rolling(20).mean()
    df["cci_20"] = (tp - sma_tp) / (0.015 * (mad + 1e-8))

    up_days = delta.clip(lower=0).rolling(14).sum()
    down_days = (-delta.clip(upper=0)).rolling(14).sum()
    df["cmo_14"] = 100.0 * _safe_div(up_days - down_days, up_days + down_days)

    bp = df["close"] - pd.concat([df["low"], df["close"].shift(1)], axis=1).min(axis=1)
    tr_uo = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    avg7 = _safe_div(bp.rolling(7).sum(), tr_uo.rolling(7).sum())
    avg14 = _safe_div(bp.rolling(14).sum(), tr_uo.rolling(14).sum())
    avg28 = _safe_div(bp.rolling(28).sum(), tr_uo.rolling(28).sum())
    df["uo"] = 100.0 * (4 * avg7 + 2 * avg14 + avg28) / 7.0

    momentum_cols = ["rsi_14", "stoch_k", "macd_hist", "cci_20", "roc_10", "cmo_14", "uo"]
    z_cols: list[str] = []
    for col in momentum_cols:
        z_name = f"{col}_z"
        df[z_name] = _rolling_z(df[col], 50)
        z_cols.append(z_name)

    df["momentum_score"] = df[z_cols].mean(axis=1)
    df["momentum_dispersion"] = df[z_cols].std(axis=1)
    df["momentum_agreement"] = (df[z_cols] > 0).sum(axis=1) / max(len(z_cols), 1)

    tr = _true_range(df)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = _safe_div(df["atr"], df["close"])
    df["atr_expansion"] = _safe_div(df["atr"], df["atr"].rolling(50).mean())

    adx, plus_di, minus_di = _compute_adx(df, 14)
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["bop"] = _safe_div(df["close"] - df["open"], df["high"] - df["low"])

    direction = np.sign(df["close"].diff()).fillna(0.0)
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_slope"] = df["obv"].diff(5)

    money_flow = tp * df["volume"]
    pos_flow = money_flow.where(tp > tp.shift(1), 0.0)
    neg_flow = money_flow.where(tp < tp.shift(1), 0.0)
    mfi_ratio = _safe_div(pos_flow.rolling(14).sum(), neg_flow.rolling(14).sum())
    df["mfi_14"] = 100.0 - (100.0 / (1.0 + mfi_ratio))

    df["donchian_upper"] = df["high"].rolling(20).max()
    df["donchian_lower"] = df["low"].rolling(20).min()
    df["donchian_width"] = _safe_div(df["donchian_upper"] - df["donchian_lower"], df["close"])
    df["donchian_breakout"] = (df["high"] > df["donchian_upper"].shift(1)).astype(float)
    df["donchian_breakdown"] = (df["low"] < df["donchian_lower"].shift(1)).astype(float)
    df["donchian_distance_up"] = _safe_div(df["close"] - df["donchian_upper"].shift(1), df["atr"] + 1e-8)
    df["donchian_distance_down"] = _safe_div(df["donchian_lower"].shift(1) - df["close"], df["atr"] + 1e-8)

    df["hour"] = df["timestamp"].dt.hour.astype(np.float32)
    df["minute"] = df["timestamp"].dt.minute.astype(np.float32)
    df["is_opening_window"] = (
        ((df["timestamp"].dt.hour == 9) & (df["timestamp"].dt.minute >= 30))
        | (df["timestamp"].dt.hour == 10)
    ).astype(np.float32)
    df["is_midday"] = ((df["timestamp"].dt.hour >= 11) & (df["timestamp"].dt.hour < 14)).astype(np.float32)
    df["is_power_hour"] = (df["timestamp"].dt.hour >= 15).astype(np.float32)

    if "pred_ret_5" not in df.columns:
        df["pred_ret_5"] = df["ret_5"].shift(-1)
    if "pred_ret_15" not in df.columns:
        df["pred_ret_15"] = df["ret_15"].shift(-1)
    if "pred_ret_30" not in df.columns:
        df["pred_ret_30"] = df["ret_30"].shift(-1)

    if "breakout_prob" not in df.columns:
        df["breakout_prob"] = (
            0.4 * df["donchian_breakout"].fillna(0.0)
            + 0.3 * (df["momentum_score"] > 0).astype(float)
            + 0.3 * (df["adx"] > df["adx"].rolling(50).mean()).astype(float)
        ).clip(0.0, 1.0)

    if "continuation_prob" not in df.columns:
        bull_cont = (
            0.5 * (df["momentum_agreement"] > 0.5).astype(float)
            + 0.5 * (df["price_vs_sma20"] > 0).astype(float)
        )
        bear_cont = (
            0.5 * (df["momentum_agreement"] < 0.5).astype(float)
            + 0.5 * (df["price_vs_sma20"] < 0).astype(float)
        )
        df["continuation_prob"] = np.maximum(bull_cont, bear_cont).clip(0.0, 1.0)

    if "signal_confidence" not in df.columns:
        df["signal_confidence"] = np.clip(
            0.35 * _safe_div(df["pred_ret_15"].abs().fillna(0.0), df["atr_pct"] + 1e-8)
            + 0.35 * df["breakout_prob"].fillna(0.0)
            + 0.30 * df["continuation_prob"].fillna(0.0),
            0.0,
            1.0,
        )

    regime_feature_cols = ["ret_1", "ret_5", "atr_pct", "adx", "price_vs_sma20"]

    if fit_regime_engine:
        regime_engine = HMMRegimeEngine().fit(
            df.dropna(subset=regime_feature_cols),
            regime_feature_cols,
        )

    if regime_engine is not None and regime_engine.is_fitted:
        df = regime_engine.predict(df)
    else:
        bull_mask = (
            (df["sma_20"] > df["sma_50"])
            & (df["price_vs_sma20"] > 0)
            & (df["sma20_slope"] > 0)
        )
        bear_mask = (
            (df["sma_20"] < df["sma_50"])
            & (df["price_vs_sma20"] < 0)
            & (df["sma20_slope"] < 0)
        )

        df["regime"] = np.where(
            bull_mask,
            "BULL",
            np.where(bear_mask, "BEAR", "TRANSITION"),
        )

        df["regime_conf"] = np.clip(df["price_vs_sma20"].abs().fillna(0.0), 0.0, 1.0)
        df["bull_score"] = np.where(df["regime"] == "BULL", df["regime_conf"], 0.0)
        df["bear_score"] = np.where(df["regime"] == "BEAR", df["regime_conf"], 0.0)
        df["transition_score"] = np.where(df["regime"] == "TRANSITION", 1.0 - df["regime_conf"], 0.0)

    df["trading_enabled"] = (df["regime"] != "TRANSITION").astype(np.float32)
    df = _build_universe_context(df)

    feature_cols_to_clean = [
        "pred_ret_5", "pred_ret_15", "pred_ret_30",
        "momentum_score", "momentum_dispersion", "momentum_agreement",
        "price_vs_sma20", "sma20_slope", "tema20_slope",
        "atr_pct", "atr_expansion", "adx",
        "obv_slope", "mfi_14", "bop",
        "donchian_width", "donchian_distance_up", "donchian_distance_down",
        "signal_confidence", "regime_conf", "bull_score", "bear_score", "transition_score",
        "tqqq_close", "sqqq_close", "qqq_close",
    ] + [c for c in df.columns if c.startswith("universe_")]

    for col in feature_cols_to_clean:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna().reset_index(drop=True)

    print("[features] feature columns check:")
    for col in ["qqq_close", "tqqq_close", "sqqq_close", "bull_cross_state", "bear_cross_state"]:
        print(f"  {col}: {'yes' if col in df.columns else 'no'}")

    universe_close_cols = [c for c in df.columns if c.endswith("_close") and c not in {"close"}]
    print(f"[features] retained universe close columns: {len(universe_close_cols)}")
    print(f"[features] retained universe context columns: {len([c for c in df.columns if c.startswith('universe_')])}")

    return df, regime_engine


class LiveTrader:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        vecnorm_path: str | Path = VECNORM_PATH,
        regime_engine_path: str | Path = REGIME_ENGINE_PATH,
    ):
        self.model = PPO.load(str(model_path))
        self.vec_norm_path = Path(vecnorm_path)
        self.vec_norm: VecNormalize | None = None

        self.regime_engine = (
            HMMRegimeEngine.load(regime_engine_path)
            if Path(regime_engine_path).exists()
            else None
        )

        # Must match the improved training env defaults.
        self.env_config = TradeCentricMDPConfig(
            initial_cash=100_000.0,
            hmax=100,
            transaction_cost_pct=0.001,
            invalid_action_penalty=0.001,
            turbulence_threshold_quantile=0.99,
            max_episode_steps=256,
            allow_fractional_clip_to_cash=True,
            reward_scale=1.0,
            min_feature_lookback=30,
            target_num_stocks=29,
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

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        built, _ = build_features(df, regime_engine=self.regime_engine, fit_regime_engine=False)
        return built

    def _make_dummy_env(self, feat: pd.DataFrame) -> TradingEnv:
        return TradingEnv(
            data=feat.head(min(len(feat), 500)).copy(),
            config=self.env_config,
        )

    def _load_vec_norm(self, feat: pd.DataFrame) -> None:
        if self.vec_norm is not None:
            return

        dummy_env = DummyVecEnv([lambda: self._make_dummy_env(feat)])

        if not self.vec_norm_path.exists():
            print(f"[live] VecNormalize file not found: {self.vec_norm_path}")
            self.vec_norm = None
            return

        try:
            self.vec_norm = VecNormalize.load(str(self.vec_norm_path), dummy_env)
            self.vec_norm.training = False
            self.vec_norm.norm_reward = False
            print("[live] loaded VecNormalize successfully.")
        except Exception as e:
            print(f"[live] VecNormalize load mismatch: {e}")
            print("[live] running without saved VecNormalize.")
            self.vec_norm = None

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = self.build_features(df)
        self._load_vec_norm(feat)

        env = TradingEnv(data=feat.copy(), config=self.env_config)
        obs, reset_info = env.reset()

        print(f"[live] rows={len(feat):,}")
        print(f"[live] tradable symbols={len(reset_info.get('stock_symbols', []))}")
        print("[live] starting replay...")

        records: list[dict[str, object]] = []
        done = False
        truncated = False
        step_idx = 0

        while not (done or truncated):
            if step_idx < 3:
                row = feat.iloc[env.idx]
                print(
                    f"[live] sample {step_idx} | ts={row['timestamp']} | "
                    f"qqq={float(row['qqq_close']):.2f} | regime={row['regime']} | "
                    f"regime_conf={float(row['regime_conf']):.2f}"
                )

            if self.vec_norm is not None:
                obs_in = self.vec_norm.normalize_obs(obs.reshape(1, -1))
            else:
                obs_in = obs.reshape(1, -1)

            action, _ = self.model.predict(obs_in, deterministic=True)
            action = np.asarray(action).reshape(-1)

            obs, reward, done, truncated, info = env.step(action)

            records.append(
                {
                    "timestamp": feat.iloc[env.idx]["timestamp"] if env.idx < len(feat) else step_idx,
                    "portfolio_value": float(info["portfolio_value"]),
                    "balance": float(info["balance"]),
                    "step_portfolio_change": float(info["step_portfolio_change"]),
                    "reward": float(reward),
                    "turbulence": float(info["turbulence"]),
                    "turbulence_threshold": float(info["turbulence_threshold"]),
                    "stock_symbols": ",".join(info["stock_symbols"]),
                    "holdings_json": pd.Series(info["holdings"], index=info["stock_symbols"]).to_json(),
                    "prices_json": pd.Series(info["prices"], index=info["stock_symbols"]).to_json(),
                    "action_json": pd.Series(action, index=info["stock_symbols"]).to_json(),
                }
            )
            step_idx += 1

        print(f"[live] replay finished | records={len(records)}")

        results = pd.DataFrame(records)
        print(f"[live] results shape: {results.shape}")
        print(f"[live] results columns: {list(results.columns)}")

        self.plot_results(results)
        return results

    def plot_results(self, results: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        if results is None or results.empty:
            print("[live] no results to plot.")
            return

        results = results.copy()
        results["timestamp"] = pd.to_datetime(results["timestamp"], errors="coerce")
        results = results.dropna(subset=["timestamp"]).reset_index(drop=True)

        if results.empty:
            print("[live] no valid timestamped results to plot.")
            return

        for col in ["portfolio_value", "balance", "step_portfolio_change", "reward", "turbulence"]:
            if col in results.columns:
                results[col] = pd.to_numeric(results[col], errors="coerce")

        results = results.dropna(subset=["portfolio_value"]).reset_index(drop=True)
        if results.empty:
            print("[live] no valid portfolio values to plot.")
            return

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(results["timestamp"], results["portfolio_value"], linewidth=2, label="Portfolio Value")
        axes[0].set_title("Trade-Centric Multi-Stock Portfolio Value")
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.3)

        axes[1].plot(results["timestamp"], results["balance"], linewidth=1.5, label="Cash Balance")
        axes[1].set_title("Cash Balance")
        axes[1].legend(loc="upper left")
        axes[1].grid(alpha=0.3)

        axes[2].plot(results["timestamp"], results["turbulence"], linewidth=1.25, label="Turbulence")
        if "turbulence_threshold" in results.columns and results["turbulence_threshold"].notna().any():
            axes[2].axhline(
                y=float(results["turbulence_threshold"].dropna().iloc[0]),
                linestyle="--",
                label="Turbulence Threshold",
            )
        axes[2].set_title("Turbulence Monitor")
        axes[2].legend(loc="upper left")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    training_data_path = LIVE_DIR / "training_data.csv"

    if not training_data_path.exists():
        raise FileNotFoundError(f"Missing data file: {training_data_path}")

    df = pd.read_csv(training_data_path)
    trader = LiveTrader()
    trader.run(df)
