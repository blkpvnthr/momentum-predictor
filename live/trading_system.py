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
    """
    Fits a Gaussian HMM to QQQ-derived market state features and maps hidden
    states to BULL / BEAR / TRANSITION.
    """

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

        if "ret_1" in temp.columns:
            ret_col = "ret_1"
        elif "ret_5" in temp.columns:
            ret_col = "ret_5"
        else:
            ret_col = self.feature_cols[0]

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
    """
    Build QQQ-driven features.

    Expected minimum input:
    - timestamp, open, high, low, close, volume   (QQQ bar data)
    - tqqq_close, sqqq_close                      (execution prices)
    """
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
    Action map must match TradingEnv:

        0 = HOLD
        1 = ENTER_TQQQ
        2 = EXIT_TQQQ
        3 = ENTER_SQQQ
        4 = EXIT_SQQQ
    """

    SYMBOL_FLAT = 0
    SYMBOL_TQQQ = 1
    SYMBOL_SQQQ = 2

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        vecnorm_path: str | Path = VECNORM_PATH,
        regime_engine_path: str | Path = REGIME_ENGINE_PATH,
        transition_entry_confidence: float = 0.85,
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

        self.position_symbol = self.SYMBOL_FLAT
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_trade = 0
        self.max_favorable_excursion = 0.0
        self.current_drawdown = 0.0

        self.initial_equity = 1.0
        self.equity = 1.0
        self.realized_pnl = 0.0
        self.prev_total_pnl = 0.0

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
            print("[live] loaded VecNormalize successfully.")
        except AssertionError as e:
            print(f"[live] VecNormalize shape mismatch: {e}")
            print("[live] running without saved VecNormalize.")
            self.vec_norm = None

    def _get_price(self, row: pd.Series) -> float:
        if self.position_symbol == self.SYMBOL_TQQQ:
            return float(row["tqqq_close"])
        if self.position_symbol == self.SYMBOL_SQQQ:
            return float(row["sqqq_close"])
        return float(row["qqq_close"])

    def _entry_price(self, row: pd.Series, symbol: int) -> float:
        if symbol == self.SYMBOL_TQQQ:
            return float(row["tqqq_close"])
        if symbol == self.SYMBOL_SQQQ:
            return float(row["sqqq_close"])
        raise ValueError(f"Unsupported symbol code: {symbol}")

    def _reset_position_only(self) -> None:
        self.position_symbol = self.SYMBOL_FLAT
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_trade = 0
        self.max_favorable_excursion = 0.0
        self.current_drawdown = 0.0

    def _reset_portfolio(self) -> None:
        self._reset_position_only()
        self.equity = self.initial_equity
        self.realized_pnl = 0.0
        self.prev_total_pnl = 0.0

    def _is_flat(self) -> bool:
        return self.position_symbol == self.SYMBOL_FLAT

    def _is_tqqq(self) -> bool:
        return self.position_symbol == self.SYMBOL_TQQQ

    def _is_sqqq(self) -> bool:
        return self.position_symbol == self.SYMBOL_SQQQ

    def _bull_signal_ok(self, row: pd.Series) -> bool:
        return bool(
            float(row["bull_score"]) >= 0.60
            and float(row["signal_confidence"]) >= 0.55
            and float(row["price_vs_sma20"]) > 0.0
            and float(row["sma20_slope"]) > 0.0
            and float(row["bull_cross_state"]) > 0.5
        )

    def _bear_signal_ok(self, row: pd.Series) -> bool:
        return bool(
            float(row["bear_score"]) >= 0.60
            and float(row["signal_confidence"]) >= 0.55
            and float(row["price_vs_sma20"]) < 0.0
            and float(row["sma20_slope"]) < 0.0
            and float(row["bear_cross_state"]) > 0.5
        )

    def _transition_allows_tqqq(self, row: pd.Series) -> bool:
        return bool(
            float(row["regime_conf"]) >= self.transition_entry_confidence
            and float(row["bull_score"]) > float(row["bear_score"])
            and self._bull_signal_ok(row)
        )

    def _transition_allows_sqqq(self, row: pd.Series) -> bool:
        return bool(
            float(row["regime_conf"]) >= self.transition_entry_confidence
            and float(row["bear_score"]) > float(row["bull_score"])
            and self._bear_signal_ok(row)
        )

    def get_obs(self, row: pd.Series) -> np.ndarray:
        base = [float(row[c]) for c in self.observation_columns()]
        controlled = [
            float(self.position_symbol == self.SYMBOL_TQQQ),
            float(self.position_symbol == self.SYMBOL_SQQQ),
            float(self.unrealized_pnl),
            float(self.steps_in_trade),
            float(self.current_drawdown),
        ]
        return np.asarray(base + controlled, dtype=np.float32)

    def select_action(self, obs: np.ndarray) -> int:
        if self.vec_norm is not None:
            obs_in = self.vec_norm.normalize_obs(obs.reshape(1, -1))
        else:
            obs_in = obs.reshape(1, -1)

        action, _ = self.model.predict(obs_in, deterministic=True)
        return int(np.asarray(action).item())

    def update_position_state(self, row: pd.Series) -> None:
        price = self._get_price(row)

        if self.position_symbol != self.SYMBOL_FLAT:
            pnl = (price - self.entry_price) / max(self.entry_price, 1e-8)
        else:
            pnl = 0.0

        self.unrealized_pnl = pnl

        if self.position_symbol != self.SYMBOL_FLAT:
            self.steps_in_trade += 1
            self.max_favorable_excursion = max(self.max_favorable_excursion, pnl)
            self.current_drawdown = self.max_favorable_excursion - pnl

    def update_equity(self) -> None:
        total_pnl = self.realized_pnl + self.unrealized_pnl
        pnl_delta = total_pnl - self.prev_total_pnl
        self.equity += pnl_delta
        self.prev_total_pnl = total_pnl

    def _forced_regime_action(self, row: pd.Series) -> str | None:
        regime = str(row["regime"])

        if self._is_tqqq() and regime == "BEAR":
            exit_pnl = self.unrealized_pnl
            self.realized_pnl += exit_pnl
            print(
                f"FORCED_EXIT_TQQQ @ {float(row['tqqq_close']):.2f} | "
                f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                f"trade_pnl={exit_pnl:.4f}"
            )
            self._reset_position_only()
            return "FORCED_EXIT_TQQQ"

        if self._is_sqqq() and regime == "BULL":
            exit_pnl = self.unrealized_pnl
            self.realized_pnl += exit_pnl
            print(
                f"FORCED_EXIT_SQQQ @ {float(row['sqqq_close']):.2f} | "
                f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                f"trade_pnl={exit_pnl:.4f}"
            )
            self._reset_position_only()
            return "FORCED_EXIT_SQQQ"

        return None

    def execute_action(self, action: int, row: pd.Series) -> str:
        """
        Returns a human-readable action label for logging/plotting.
        Opposite-side holdings are liquidated on hard regime flips.
        Transition entries are limited by regime confidence.
        """
        forced = self._forced_regime_action(row)
        if forced is not None:
            return forced

        regime = str(row["regime"])

        if self._is_flat():
            if action == 1:
                if regime == "BULL" and self._bull_signal_ok(row):
                    self.position_symbol = self.SYMBOL_TQQQ
                    self.entry_price = self._entry_price(row, self.SYMBOL_TQQQ)
                    self.unrealized_pnl = 0.0
                    self.steps_in_trade = 0
                    self.max_favorable_excursion = 0.0
                    self.current_drawdown = 0.0
                    print(
                        f"ENTER_TQQQ @ {float(row['tqqq_close']):.2f} | "
                        f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                        f"conf={float(row['regime_conf']):.2f}"
                    )
                    return "ENTER_TQQQ"

                if regime == "TRANSITION" and self._transition_allows_tqqq(row):
                    self.position_symbol = self.SYMBOL_TQQQ
                    self.entry_price = self._entry_price(row, self.SYMBOL_TQQQ)
                    self.unrealized_pnl = 0.0
                    self.steps_in_trade = 0
                    self.max_favorable_excursion = 0.0
                    self.current_drawdown = 0.0
                    print(
                        f"ENTER_TQQQ_TRANSITION @ {float(row['tqqq_close']):.2f} | "
                        f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                        f"conf={float(row['regime_conf']):.2f}"
                    )
                    return "ENTER_TQQQ_TRANSITION"

            if action == 3:
                if regime == "BEAR" and self._bear_signal_ok(row):
                    self.position_symbol = self.SYMBOL_SQQQ
                    self.entry_price = self._entry_price(row, self.SYMBOL_SQQQ)
                    self.unrealized_pnl = 0.0
                    self.steps_in_trade = 0
                    self.max_favorable_excursion = 0.0
                    self.current_drawdown = 0.0
                    print(
                        f"ENTER_SQQQ @ {float(row['sqqq_close']):.2f} | "
                        f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                        f"conf={float(row['regime_conf']):.2f}"
                    )
                    return "ENTER_SQQQ"

                if regime == "TRANSITION" and self._transition_allows_sqqq(row):
                    self.position_symbol = self.SYMBOL_SQQQ
                    self.entry_price = self._entry_price(row, self.SYMBOL_SQQQ)
                    self.unrealized_pnl = 0.0
                    self.steps_in_trade = 0
                    self.max_favorable_excursion = 0.0
                    self.current_drawdown = 0.0
                    print(
                        f"ENTER_SQQQ_TRANSITION @ {float(row['sqqq_close']):.2f} | "
                        f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                        f"conf={float(row['regime_conf']):.2f}"
                    )
                    return "ENTER_SQQQ_TRANSITION"

        if action == 2 and self._is_tqqq():
            exit_pnl = self.unrealized_pnl
            self.realized_pnl += exit_pnl
            print(
                f"EXIT_TQQQ @ {float(row['tqqq_close']):.2f} | "
                f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                f"trade_pnl={exit_pnl:.4f}"
            )
            self._reset_position_only()
            return "EXIT_TQQQ"

        if action == 4 and self._is_sqqq():
            exit_pnl = self.unrealized_pnl
            self.realized_pnl += exit_pnl
            print(
                f"EXIT_SQQQ @ {float(row['sqqq_close']):.2f} | "
                f"qqq={float(row['qqq_close']):.2f} | regime={regime} | "
                f"trade_pnl={exit_pnl:.4f}"
            )
            self._reset_position_only()
            return "EXIT_SQQQ"

        return "HOLD"

    def run(self, df: pd.DataFrame) -> None:
        feat = self.build_features(df)
        self._load_vec_norm(feat)
        self._reset_portfolio()

        print(f"[live] rows={len(feat):,}")
        print("[live] starting replay...")

        records: list[dict[str, object]] = []

        for i, row in feat.iterrows():
            if i < 3:
                print(
                    f"[live] sample {i} | ts={row['timestamp']} | "
                    f"qqq={float(row['qqq_close']):.2f} | regime={row['regime']} | "
                    f"regime_conf={float(row['regime_conf']):.2f}"
                )

            self.update_position_state(row)
            obs = self.get_obs(row)
            action = self.select_action(obs)
            action_label = self.execute_action(action, row)

            # Recompute after potential entry/exit/forced liquidation
            self.update_position_state(row)
            self.update_equity()

            records.append(
                {
                    "timestamp": row["timestamp"],
                    "qqq_close": float(row["qqq_close"]),
                    "tqqq_close": float(row["tqqq_close"]),
                    "sqqq_close": float(row["sqqq_close"]),
                    "action": int(action),
                    "action_label": action_label,
                    "position_symbol": int(self.position_symbol),
                    "unrealized_pnl": float(self.unrealized_pnl),
                    "realized_pnl": float(self.realized_pnl),
                    "equity": float(self.equity),
                    "regime": str(row["regime"]),
                    "regime_conf": float(row["regime_conf"]),
                    "signal_confidence": float(row["signal_confidence"]),
                    "bull_score": float(row["bull_score"]),
                    "bear_score": float(row["bear_score"]),
                }
            )

        print(f"[live] replay finished | records={len(records)}")

        results = pd.DataFrame(records)
        print(f"[live] results shape: {results.shape}")
        print(f"[live] results columns: {list(results.columns)}")

        self.plot_results(results)

    def plot_results(self, results: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        if results is None:
            print("[live] results is None.")
            return

        if results.empty:
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

        for col in ["qqq_close", "tqqq_close", "sqqq_close", "equity"]:
            if col in results.columns:
                results[col] = pd.to_numeric(results[col], errors="coerce")

        results = results.dropna(subset=["qqq_close", "equity"]).reset_index(drop=True)
        if results.empty:
            print("[live] no valid data to plot.")
            return

        initial_equity = self.initial_equity
        results["strategy_equity"] = results["equity"]

        qqq_start = float(results["qqq_close"].iloc[0])
        results["qqq_equity"] = initial_equity * (results["qqq_close"] / max(qqq_start, 1e-8))

        # Normalized strategy equity for overlay on QQQ price panel
        strategy_start = float(results["strategy_equity"].iloc[0])
        results["strategy_equity_norm"] = results["strategy_equity"] / max(strategy_start, 1e-8)

        tqqq_buy_idx = results["action_label"].isin(["ENTER_TQQQ", "ENTER_TQQQ_TRANSITION"])
        tqqq_sell_idx = results["action_label"].isin(["EXIT_TQQQ", "FORCED_EXIT_TQQQ"])
        sqqq_buy_idx = results["action_label"].isin(["ENTER_SQQQ", "ENTER_SQQQ_TRANSITION"])
        sqqq_sell_idx = results["action_label"].isin(["EXIT_SQQQ", "FORCED_EXIT_SQQQ"])

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # -------------------------------------------------
        # Top: QQQ price + normalized strategy equity overlay
        # -------------------------------------------------
        ax_price = axes[0]
        ax_price.plot(
            results["timestamp"],
            results["qqq_close"],
            linewidth=1.5,
            label="QQQ Price",
        )
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

        # -------------------------------------------------
        # Middle: Strategy equity vs QQQ equity
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Bottom: Trade actions over QQQ
        # -------------------------------------------------
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

        axes[2].set_title("Trade Actions on QQQ Price")
        axes[2].legend(loc="upper left", ncol=2)
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