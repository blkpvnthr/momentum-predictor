from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trading_env import TradingEnv

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


# =========================================================
# REGIME ENGINE
# =========================================================
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


# =========================================================
# FEATURE ENGINEERING
# =========================================================
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

    feature_cols_to_clean = [
        "pred_ret_5",
        "pred_ret_15",
        "pred_ret_30",
        "momentum_score",
        "momentum_dispersion",
        "momentum_agreement",
        "price_vs_sma20",
        "sma20_slope",
        "tema20_slope",
        "atr_pct",
        "atr_expansion",
        "adx",
        "obv_slope",
        "mfi_14",
        "bop",
        "donchian_width",
        "donchian_distance_up",
        "donchian_distance_down",
        "signal_confidence",
        "regime_conf",
        "bull_score",
        "bear_score",
        "transition_score",
        "tqqq_close",
        "sqqq_close",
        "qqq_close",
    ]

    for col in feature_cols_to_clean:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna().reset_index(drop=True)

    print("[features] feature columns check:")
    for col in ["qqq_close", "tqqq_close", "sqqq_close", "bull_cross_state", "bear_cross_state"]:
        print(f"  {col}: {'yes' if col in df.columns else 'no'}")

    return df, regime_engine


# =========================================================
# LIVE TRADER
# =========================================================
class LiveTrader:
    """
    9-action runtime map aligned with the safer environment.

        0 = HOLD
        1 = ENTER_TQQQ_10
        2 = ENTER_TQQQ_20
        3 = ENTER_TQQQ_35
        4 = EXIT_TQQQ
        5 = ENTER_SQQQ_10
        6 = ENTER_SQQQ_20
        7 = ENTER_SQQQ_35
        8 = EXIT_SQQQ
    """

    SYMBOL_FLAT = 0
    SYMBOL_TQQQ = 1
    SYMBOL_SQQQ = 2

    TQQQ_ENTRY_ACTIONS = {1: 0.10, 2: 0.15}
    SQQQ_ENTRY_ACTIONS = {5: 0.10, 6: 0.15}

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        vecnorm_path: str | Path = VECNORM_PATH,
        regime_engine_path: str | Path = REGIME_ENGINE_PATH,
        transition_entry_confidence: float = 0.97,
        cooldown_bars: int = 30,
        score_edge_threshold: float = 0.22,
        min_bull_score: float = 0.72,
        min_bear_score: float = 0.72,
        min_signal_confidence: float = 0.72,
        min_hold_bars: int = 15,
        initial_equity: float = 100_000.0,
        fee_rate: float = 0.0015,
        slippage_bps: float = 5.0,
    ):
        self.model = PPO.load(str(model_path))
        self.vec_norm_path = Path(vecnorm_path)
        self.vec_norm: VecNormalize | None = None

        self.regime_engine = (
            HMMRegimeEngine.load(regime_engine_path)
            if Path(regime_engine_path).exists()
            else None
        )

        self.transition_entry_confidence = float(transition_entry_confidence)
        self.cooldown_bars = int(cooldown_bars)
        self.score_edge_threshold = float(score_edge_threshold)
        self.min_bull_score = float(min_bull_score)
        self.min_bear_score = float(min_bear_score)
        self.min_signal_confidence = float(min_signal_confidence)
        self.min_hold_bars = int(min_hold_bars)

        self.initial_equity = float(initial_equity)
        self.fee_rate = float(fee_rate)
        self.slippage_bps = float(slippage_bps)

        self.position_symbol = self.SYMBOL_FLAT
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_cost_basis = 0.0
        self.shares = 0
        self.market_value = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_trade = 0
        self.max_favorable_excursion = 0.0
        self.current_drawdown = 0.0

        self.cash = self.initial_equity
        self.equity = self.initial_equity
        self.realized_pnl = 0.0

        self.last_trade_step = -10_000
        self.global_step = 0

    @staticmethod
    def observation_columns() -> list[str]:
        return [
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

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        built, _ = build_features(df, regime_engine=self.regime_engine, fit_regime_engine=False)
        return built

    def _load_vec_norm(self, feat: pd.DataFrame) -> None:
        if self.vec_norm is not None:
            return

        dummy_env = DummyVecEnv([lambda: TradingEnv(feat.head(500).copy())])

        if not self.vec_norm_path.exists():
            print(f"[live] VecNormalize file not found: {self.vec_norm_path}")
            self.vec_norm = None
            return

        try:
            self.vec_norm = VecNormalize.load(str(self.vec_norm_path), dummy_env)
            self.vec_norm.training = False
            self.vec_norm.norm_reward = False

            sample_obs = self.get_obs(feat.iloc[0]).reshape(1, -1)
            expected_dim = int(np.asarray(self.vec_norm.obs_rms.mean).shape[0])
            actual_dim = int(sample_obs.shape[1])

            if actual_dim != expected_dim:
                print(
                    f"[live] VecNormalize shape mismatch at load time | "
                    f"expected={expected_dim} actual={actual_dim}"
                )
                print("[live] ignoring saved VecNormalize and running without it.")
                self.vec_norm = None
                return

            print("[live] loaded VecNormalize successfully.")
        except Exception as exc:
            print(f"[live] failed to load VecNormalize: {exc}")
            print("[live] running without saved VecNormalize.")
            self.vec_norm = None

    def _reset_position_only(self) -> None:
        self.position_symbol = self.SYMBOL_FLAT
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_cost_basis = 0.0
        self.shares = 0
        self.market_value = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_trade = 0
        self.max_favorable_excursion = 0.0
        self.current_drawdown = 0.0

    def _reset_portfolio(self) -> None:
        self._reset_position_only()
        self.cash = self.initial_equity
        self.equity = self.initial_equity
        self.realized_pnl = 0.0
        self.last_trade_step = -10_000
        self.global_step = 0

    def _is_flat(self) -> bool:
        return self.position_symbol == self.SYMBOL_FLAT or self.shares <= 0

    def _is_tqqq(self) -> bool:
        return self.position_symbol == self.SYMBOL_TQQQ and self.shares > 0

    def _is_sqqq(self) -> bool:
        return self.position_symbol == self.SYMBOL_SQQQ and self.shares > 0

    def _in_cooldown(self) -> bool:
        return (self.global_step - self.last_trade_step) < self.cooldown_bars

    def _mark_price(self, row: pd.Series, symbol: int | None = None) -> float:
        symbol = self.position_symbol if symbol is None else symbol
        if symbol == self.SYMBOL_TQQQ:
            return float(row["tqqq_close"])
        if symbol == self.SYMBOL_SQQQ:
            return float(row["sqqq_close"])
        return float(row["qqq_close"])

    def _fill_price(self, row: pd.Series, symbol: int, is_buy: bool) -> float:
        mid = self._mark_price(row, symbol)
        slip = self.slippage_bps / 10_000.0
        return mid * (1.0 + slip) if is_buy else mid * (1.0 - slip)

    def _bull_signal_score(self, row: pd.Series) -> float:
        score = 0.0
        score += 0.35 * float(row["bull_score"])
        score += 0.20 * float(row["signal_confidence"])
        score += 0.15 * float(float(row["price_vs_sma20"]) > 0.0)
        score += 0.10 * float(float(row["sma20_slope"]) > 0.0)
        score += 0.10 * float(float(row["bull_cross_state"]) > 0.5)
        score += 0.10 * float(float(row["momentum_score"]) > 0.0)
        return float(score)

    def _bear_signal_score(self, row: pd.Series) -> float:
        score = 0.0
        score += 0.35 * float(row["bear_score"])
        score += 0.20 * float(row["signal_confidence"])
        score += 0.15 * float(float(row["price_vs_sma20"]) < 0.0)
        score += 0.10 * float(float(row["sma20_slope"]) < 0.0)
        score += 0.10 * float(float(row["bear_cross_state"]) > 0.5)
        score += 0.10 * float(float(row["momentum_score"]) < 0.0)
        return float(score)

    def _score_edge(self, row: pd.Series) -> float:
        return self._bull_signal_score(row) - self._bear_signal_score(row)

    def _bull_signal_ok(self, row: pd.Series) -> bool:
        edge = self._score_edge(row)
        return bool(
            float(row["bull_score"]) >= self.min_bull_score
            and float(row["signal_confidence"]) >= self.min_signal_confidence
            and float(row["price_vs_sma20"]) > 0.0
            and float(row["sma20_slope"]) > 0.0
            and float(row["bull_cross_state"]) > 0.5
            and float(row["momentum_score"]) >= 0.0
            and edge >= self.score_edge_threshold
        )

    def _bear_signal_ok(self, row: pd.Series) -> bool:
        edge = self._score_edge(row)
        return bool(
            float(row["bear_score"]) >= self.min_bear_score
            and float(row["signal_confidence"]) >= self.min_signal_confidence
            and float(row["price_vs_sma20"]) < 0.0
            and float(row["sma20_slope"]) < 0.0
            and float(row["bear_cross_state"]) > 0.5
            and float(row["momentum_score"]) <= 0.0
            and edge <= -self.score_edge_threshold
        )

    def get_obs(self, row: pd.Series) -> np.ndarray:
        base = [float(row[c]) for c in self.observation_columns()]
        controlled = [
            float(self.position_symbol == self.SYMBOL_TQQQ),
            float(self.position_symbol == self.SYMBOL_SQQQ),
            float(self.position_size),
            float(self.unrealized_pnl / max(self.initial_equity, 1e-8)),
            float(self.steps_in_trade),
            float(self.current_drawdown / max(self.initial_equity, 1e-8)),
        ]
        return np.asarray(base + controlled, dtype=np.float32)

    def select_action(self, obs: np.ndarray) -> int:
        obs_in = obs.reshape(1, -1)

        if self.vec_norm is not None:
            try:
                expected_dim = int(np.asarray(self.vec_norm.obs_rms.mean).shape[0])
                actual_dim = int(obs_in.shape[1])

                if actual_dim == expected_dim:
                    obs_in = self.vec_norm.normalize_obs(obs_in)
                else:
                    print(
                        f"[live] VecNormalize dimension mismatch | "
                        f"expected={expected_dim} actual={actual_dim} | "
                        f"running without normalization."
                    )
                    self.vec_norm = None
            except Exception as exc:
                print(f"[live] VecNormalize normalization failed: {exc}")
                print("[live] running without normalization.")
                self.vec_norm = None

        action, _ = self.model.predict(obs_in, deterministic=True)
        return int(np.asarray(action).item())

    def update_position_state(self, row: pd.Series) -> None:
        if self._is_flat():
            self.market_value = 0.0
            self.unrealized_pnl = 0.0
            self.equity = self.cash
            return

        mark = self._mark_price(row)
        self.market_value = float(self.shares) * mark
        self.unrealized_pnl = self.market_value - self.entry_cost_basis
        self.equity = self.cash + self.market_value

        self.steps_in_trade += 1
        self.max_favorable_excursion = max(self.max_favorable_excursion, self.unrealized_pnl)
        self.current_drawdown = self.max_favorable_excursion - self.unrealized_pnl

    def _open_position(self, row: pd.Series, symbol: int, size: float) -> bool:
        if not self._is_flat():
            return False

        target_fraction = float(np.clip(size, 0.0, 1.0))
        if target_fraction <= 0.0:
            return False

        fill_price = self._fill_price(row, symbol, is_buy=True)
        desired_notional = self.equity * target_fraction
        budget = min(self.cash, desired_notional)

        if budget <= 0.0:
            return False

        per_share_total_cost = fill_price * (1.0 + self.fee_rate)
        shares = int(budget // per_share_total_cost)

        if shares <= 0:
            return False

        gross_cost = shares * fill_price
        fees = gross_cost * self.fee_rate
        total_cost = gross_cost + fees

        self.cash -= total_cost
        self.position_symbol = symbol
        self.position_size = target_fraction
        self.entry_price = fill_price
        self.entry_cost_basis = total_cost
        self.shares = shares
        self.market_value = shares * fill_price
        self.unrealized_pnl = self.market_value - self.entry_cost_basis
        self.steps_in_trade = 0
        self.max_favorable_excursion = 0.0
        self.current_drawdown = 0.0
        self.last_trade_step = self.global_step
        self.equity = self.cash + self.market_value
        return True

    def _close_position(self, row: pd.Series) -> float:
        if self._is_flat():
            return 0.0

        fill_price = self._fill_price(row, self.position_symbol, is_buy=False)
        gross_proceeds = self.shares * fill_price
        fees = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - fees

        trade_realized_pnl = net_proceeds - self.entry_cost_basis
        self.cash += net_proceeds
        self.realized_pnl += trade_realized_pnl

        self._reset_position_only()
        self.last_trade_step = self.global_step
        self.equity = self.cash
        return trade_realized_pnl

    def _forced_regime_action(self, row: pd.Series) -> str | None:
        regime = str(row["regime"])

        if self._is_tqqq() and regime == "BEAR":
            size_pct = int(round(self.position_size * 100))
            exit_pnl = self._close_position(row)
            print(
                f"FORCED_EXIT_TQQQ_{size_pct} @ {float(row['tqqq_close']):.2f} | "
                f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                f"trade_pnl=${exit_pnl:,.2f}"
            )
            return f"FORCED_EXIT_TQQQ_{size_pct}"

        if self._is_sqqq() and regime == "BULL":
            size_pct = int(round(self.position_size * 100))
            exit_pnl = self._close_position(row)
            print(
                f"FORCED_EXIT_SQQQ_{size_pct} @ {float(row['sqqq_close']):.2f} | "
                f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                f"trade_pnl=${exit_pnl:,.2f}"
            )
            return f"FORCED_EXIT_SQQQ_{size_pct}"

        return None

    def execute_action(self, action: int, row: pd.Series) -> str:
        forced = self._forced_regime_action(row)
        if forced is not None:
            return forced

        if self._in_cooldown():
            return "HOLD_COOLDOWN"

        regime = str(row["regime"])
        edge = self._score_edge(row)

        if self._is_flat():
            if action in self.TQQQ_ENTRY_ACTIONS:
                requested_size = self.TQQQ_ENTRY_ACTIONS[action]

                if float(row["bear_score"]) >= float(row["bull_score"]) - 0.03:
                    return "HOLD"

                if regime == "BULL" and self._bull_signal_ok(row):
                    ok = self._open_position(row, self.SYMBOL_TQQQ, requested_size)
                    if ok:
                        label = f"ENTER_TQQQ_{int(round(requested_size * 100))}"
                        print(
                            f"{label} @ {float(row['tqqq_close']):.2f} | "
                            f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                            f"conf={float(row['regime_conf']):.2f} | edge={edge:.3f} | "
                            f"shares={self.shares} | cash=${self.cash:,.2f}"
                        )
                        return label
                    return "HOLD"

            if action in self.SQQQ_ENTRY_ACTIONS:
                requested_size = self.SQQQ_ENTRY_ACTIONS[action]

                if float(row["bull_score"]) >= float(row["bear_score"]) - 0.03:
                    return "HOLD"

                if regime == "BEAR" and self._bear_signal_ok(row):
                    ok = self._open_position(row, self.SYMBOL_SQQQ, requested_size)
                    if ok:
                        label = f"ENTER_SQQQ_{int(round(requested_size * 100))}"
                        print(
                            f"{label} @ {float(row['sqqq_close']):.2f} | "
                            f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                            f"conf={float(row['regime_conf']):.2f} | edge={edge:.3f} | "
                            f"shares={self.shares} | cash=${self.cash:,.2f}"
                        )
                        return label
                    return "HOLD"

        if action == 4 and self._is_tqqq():
            if self.steps_in_trade < self.min_hold_bars:
                return "HOLD"

            size_pct = int(round(self.position_size * 100))
            exit_pnl = self._close_position(row)
            print(
                f"EXIT_TQQQ_{size_pct} @ {float(row['tqqq_close']):.2f} | "
                f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                f"trade_pnl=${exit_pnl:,.2f} | cash=${self.cash:,.2f}"
            )
            return f"EXIT_TQQQ_{size_pct}"

        if action == 8 and self._is_sqqq():
            if self.steps_in_trade < self.min_hold_bars:
                return "HOLD"

            size_pct = int(round(self.position_size * 100))
            exit_pnl = self._close_position(row)
            print(
                f"EXIT_SQQQ_{size_pct} @ {float(row['sqqq_close']):.2f} | "
                f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                f"trade_pnl=${exit_pnl:,.2f} | cash=${self.cash:,.2f}"
            )
            return f"EXIT_SQQQ_{size_pct}"

        return "HOLD"

    def _print_performance(self, results: pd.DataFrame) -> None:
        print("\n================ PERFORMANCE ================")

        initial_equity = float(self.initial_equity)
        final_equity = float(results["equity"].iloc[-1])
        total_return = (final_equity / max(initial_equity, 1e-8)) - 1.0

        returns = pd.to_numeric(results["equity"], errors="coerce").pct_change().fillna(0.0)

        if float(returns.std()) > 0:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(252 * 390))
        else:
            sharpe = 0.0

        cum_max = pd.to_numeric(results["equity"], errors="coerce").cummax()
        drawdown = (results["equity"] - cum_max) / cum_max.replace(0.0, np.nan)
        max_dd = float(drawdown.min()) if len(drawdown) else 0.0

        trade_pnls: list[float] = []
        prev_realized = 0.0
        for pnl in pd.to_numeric(results["realized_pnl"], errors="coerce").fillna(0.0):
            delta = float(pnl) - float(prev_realized)
            if abs(delta) > 1e-9:
                trade_pnls.append(delta)
            prev_realized = float(pnl)

        wins = [p for p in trade_pnls if p > 0]
        win_rate = (len(wins) / len(trade_pnls)) if trade_pnls else 0.0

        print(f"initial_equity   : ${initial_equity:,.2f}")
        print(f"final_equity     : ${final_equity:,.2f}")
        print(f"total_return     : {total_return:.4%}")
        print(f"realized_pnl     : ${self.realized_pnl:,.2f}")
        print(f"cash             : ${self.cash:,.2f}")
        print(f"sharpe           : {sharpe:.4f}")
        print(f"max_drawdown     : {max_dd:.4%}")
        print(f"trades           : {len(trade_pnls)}")
        print(f"win_rate         : {win_rate:.2%}")

        print("\n================ REGIME DISTRIBUTION ================")
        print(results["regime"].value_counts(normalize=True).to_string())

        print("\n================ ACTION DISTRIBUTION ================")
        print(results["action_label"].value_counts().head(20).to_string())

        print("\n================ EXPOSURE ================")
        tqqq_exposure = float((results["position_symbol"] == self.SYMBOL_TQQQ).mean())
        sqqq_exposure = float((results["position_symbol"] == self.SYMBOL_SQQQ).mean())
        print(f"TQQQ exposure    : {tqqq_exposure:.2%}")
        print(f"SQQQ exposure    : {sqqq_exposure:.2%}")

        print("\n================ SCORE DIAGNOSTICS ================")
        bull_mean = float(pd.to_numeric(results["bull_score"], errors="coerce").mean())
        bear_mean = float(pd.to_numeric(results["bear_score"], errors="coerce").mean())
        print(f"mean bull_score  : {bull_mean:.4f}")
        print(f"mean bear_score  : {bear_mean:.4f}")

        print("====================================================\n")

    def run(self, df: pd.DataFrame) -> None:
        feat = self.build_features(df)
        self._load_vec_norm(feat)
        self._reset_portfolio()

        print(f"[live] rows={len(feat):,}")
        print("[live] starting replay...")

        records: list[dict[str, object]] = []

        for i, row in feat.iterrows():
            self.global_step = i

            if i < 3:
                print(
                    f"[live] sample {i} | ts={row['timestamp']} | "
                    f"qqq={float(row['qqq_close']):.2f} | regime={row['regime']} | "
                    f"regime_conf={float(row['regime_conf']):.2f}"
                )

            self.update_position_state(row)
            obs = self.get_obs(row)
            action = self.select_action(obs)

            if i < 50:
                print(
                    f"[live] step={i} regime={row['regime']} "
                    f"bull={float(row['bull_score']):.2f} "
                    f"bear={float(row['bear_score']):.2f} "
                    f"edge={self._score_edge(row):.3f} "
                    f"conf={float(row['signal_confidence']):.2f} "
                    f"action={action}"
                )

            action_label = self.execute_action(action, row)
            self.update_position_state(row)

            records.append(
                {
                    "timestamp": row["timestamp"],
                    "qqq_close": float(row["qqq_close"]),
                    "tqqq_close": float(row["tqqq_close"]),
                    "sqqq_close": float(row["sqqq_close"]),
                    "action": int(action),
                    "action_label": str(action_label),
                    "position_symbol": int(self.position_symbol),
                    "position_size": float(self.position_size),
                    "shares": int(self.shares),
                    "cash": float(self.cash),
                    "market_value": float(self.market_value),
                    "unrealized_pnl": float(self.unrealized_pnl),
                    "realized_pnl": float(self.realized_pnl),
                    "equity": float(self.equity),
                    "regime": str(row["regime"]),
                    "regime_conf": float(row["regime_conf"]),
                    "signal_confidence": float(row["signal_confidence"]),
                    "bull_score": float(row["bull_score"]),
                    "bear_score": float(row["bear_score"]),
                    "transition_score": float(row["transition_score"]),
                }
            )

        print(f"[live] replay finished | records={len(records)}")

        results = pd.DataFrame(records)
        print(f"[live] results shape: {results.shape}")
        print(f"[live] results columns: {list(results.columns)}")

        if results.empty:
            print("[live] no results.")
            return

        self._print_performance(results)
        self.plot_results(results)

    def plot_results(self, results: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        if results is None or results.empty:
            print("[live] no results to plot.")
            return

        required_cols = ["timestamp", "qqq_close", "action_label", "equity"]
        missing = [col for col in required_cols if col not in results.columns]
        if missing:
            print(f"[live] results missing required columns: {missing}")
            return

        results = results.copy()
        results["timestamp"] = pd.to_datetime(results["timestamp"], errors="coerce")
        results = results.dropna(subset=["timestamp"]).reset_index(drop=True)

        if results.empty:
            print("[live] results became empty after timestamp parsing.")
            return

        for col in ["qqq_close", "tqqq_close", "sqqq_close", "equity", "position_size", "cash", "market_value"]:
            if col in results.columns:
                results[col] = pd.to_numeric(results[col], errors="coerce")

        results = results.dropna(subset=["qqq_close", "equity"]).reset_index(drop=True)
        if results.empty:
            print("[live] no valid data to plot.")
            return

        results["strategy_equity"] = results["equity"]

        qqq_start = float(results["qqq_close"].iloc[0])
        results["qqq_equity"] = self.initial_equity * (results["qqq_close"] / max(qqq_start, 1e-8))

        strategy_start = float(results["strategy_equity"].iloc[0])
        results["strategy_equity_norm"] = results["strategy_equity"] / max(strategy_start, 1e-8)

        tqqq_buy_idx = results["action_label"].astype(str).str.contains("ENTER_TQQQ", na=False)
        tqqq_sell_idx = results["action_label"].astype(str).str.contains("EXIT_TQQQ", na=False)
        sqqq_buy_idx = results["action_label"].astype(str).str.contains("ENTER_SQQQ", na=False)
        sqqq_sell_idx = results["action_label"].astype(str).str.contains("EXIT_SQQQ", na=False)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        ax_price = axes[0]
        ax_price.plot(results["timestamp"], results["qqq_close"], linewidth=1.5, label="QQQ Price")
        ax_price.set_title("QQQ Price Action with Simulated Equity Overlay")
        ax_price.grid(alpha=0.3)

        ax_equity_overlay = ax_price.twinx()
        ax_equity_overlay.plot(
            results["timestamp"],
            results["strategy_equity_norm"],
            linewidth=1.5,
            alpha=0.85,
            label="Strategy Equity (normalized)",
        )

        price_lines, price_labels = ax_price.get_legend_handles_labels()
        eq_lines, eq_labels = ax_equity_overlay.get_legend_handles_labels()
        ax_price.legend(price_lines + eq_lines, price_labels + eq_labels, loc="upper left")

        axes[1].plot(
            results["timestamp"],
            results["strategy_equity"],
            linewidth=2,
            label="Strategy Equity",
        )
        axes[1].plot(
            results["timestamp"],
            results["qqq_equity"],
            linewidth=2,
            label="QQQ Equity",
        )
        axes[1].set_title("Simulated Strategy Equity vs QQQ Equity")
        axes[1].legend(loc="upper left")
        axes[1].grid(alpha=0.3)

        axes[2].plot(
            results["timestamp"],
            results["qqq_close"],
            linewidth=1.0,
            alpha=0.55,
            label="QQQ Price",
        )

        axes[2].scatter(
            results.loc[tqqq_buy_idx, "timestamp"],
            results.loc[tqqq_buy_idx, "qqq_close"],
            s=20,
            label="TQQQ Entry",
        )
        axes[2].scatter(
            results.loc[tqqq_sell_idx, "timestamp"],
            results.loc[tqqq_sell_idx, "qqq_close"],
            s=20,
            label="TQQQ Exit",
        )
        axes[2].scatter(
            results.loc[sqqq_buy_idx, "timestamp"],
            results.loc[sqqq_buy_idx, "qqq_close"],
            s=20,
            label="SQQQ Entry",
        )
        axes[2].scatter(
            results.loc[sqqq_sell_idx, "timestamp"],
            results.loc[sqqq_sell_idx, "qqq_close"],
            s=20,
            label="SQQQ Exit",
        )

        ax_size = axes[2].twinx()
        ax_size.plot(
            results["timestamp"],
            results["position_size"],
            linewidth=1.2,
            alpha=0.7,
            label="Position Size",
        )
        ax_size.set_ylim(-0.05, 1.05)

        lines_a, labels_a = axes[2].get_legend_handles_labels()
        lines_b, labels_b = ax_size.get_legend_handles_labels()
        axes[2].legend(lines_a + lines_b, labels_a + labels_b, loc="upper left", ncol=3)

        axes[2].set_title("Trade Actions on QQQ Price")
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