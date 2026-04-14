from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# =========================================================
# ENV / PATHS
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# =========================================================
# CONFIG
# =========================================================
MONITOR_SYMBOLS = [
    "SPY",
    "SPXU",
    "DIA",
    "DOG",
    "QQQ",
    "SQQQ",
    "TQQQ",
    "VIXY",
    "IWM",
]

INVERSE_ETF_UNIVERSE = [
    "SDOW",
    "SQQQ",
    "SRTY",
    "REW",
    "SOXS",
    "SPXU",
    "DOG",
    "DXD",
    "NVD",
]

NORMAL_UNIVERSE = [
    "SPY",
    "DIA",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "TQQQ",
    "QTUM",
    "UNG",
    "AAPL",
    "NVDA",
    "AMZN",
    "OUST",
    "IONQ",
    "QBTS",
    "LUNR",
    "SOUN",
    "UBER",
    "OPEN",
    "KXIN",
    "NGD",
    "NKTR",
    "IAG",
    "ASRT",
    "KITT",
    "GORO",
    "TECL",
    "SOXL",
    "SCHD",
    "JEPQ",
    "BUZZ",
    "AMOM",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "regime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "market_regime_analysis.csv"
STATE_JSON = OUTPUT_DIR / "market_regime_state.json"

TIMEZONE = "America/New_York"
LOOKBACK_BARS = 180
VOL_WINDOW = 20
TREND_WINDOW = 15
ZSCORE_WINDOW = 20

BULL_THRESHOLD = 0.15
BEAR_THRESHOLD = 0.15
MIN_CONFIDENCE = 0.20
TRANSITION_GAP_THRESHOLD = 0.10

REVERSAL_CONFIRM_COUNT = 3
FAILED_BREAKOUT_BUFFER = 0.0005  # 5 bps above/below prior level counts as attempt


# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class SymbolAnalysis:
    symbol: str
    last_close: float
    last_open: float
    last_high: float
    last_low: float
    prev_high_30: float
    prev_low_30: float
    ret_5: float
    ret_15: float
    ret_30: float
    vol_20: float
    slope_15: float
    zscore_20: float
    breakout_attempted_prev_high: int
    failed_breakout_prev_high: int
    breakdown_attempted_prev_low: int
    failed_breakdown_prev_low: int
    directional_score: float
    pressure_score: float


@dataclass
class RegimeResult:
    timestamp: str
    inferred_regime: str
    active_regime: str
    confidence: float
    bull_score: float
    bear_score: float
    transition_score: float
    score_gap: float
    reversal_watch: int
    candidate_regime: str
    candidate_count: int
    flip_confirmed: int
    selected_universe: str
    trading_enabled: int
    universe_size: int
    universe_symbols: str
    regime_inferred: int
    regime_active: int
    selected_universe_num: int


# =========================================================
# STATE
# =========================================================
def default_state() -> dict:
    return {
        "active_regime": "TRANSITION",
        "candidate_regime": "",
        "candidate_count": 0,
    }


def load_state(path: Path = STATE_JSON) -> dict:
    if not path.exists():
        return default_state()

    try:
        state = json.loads(path.read_text())
        if not isinstance(state, dict):
            return default_state()
        return {
            "active_regime": state.get("active_regime", "TRANSITION"),
            "candidate_regime": state.get("candidate_regime", ""),
            "candidate_count": int(state.get("candidate_count", 0)),
        }
    except Exception:
        return default_state()


def save_state(state: dict, path: Path = STATE_JSON) -> None:
    path.write_text(json.dumps(state, indent=2))


# =========================================================
# ENUM HELPERS
# =========================================================
def regime_to_num(regime: str) -> int:
    if regime == "BULL":
        return 1
    if regime == "BEAR":
        return -1
    return 0


def universe_to_num(universe: str) -> int:
    if universe == "NORMAL":
        return 1
    if universe == "INVERSE_ETF":
        return -1
    return 0


# =========================================================
# ALPACA
# =========================================================
def get_client() -> StockHistoricalDataClient:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your .env."
        )

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def fetch_bars(
    client: StockHistoricalDataClient,
    symbols: List[str],
    lookback_bars: int = LOOKBACK_BARS,
) -> pd.DataFrame:
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=10)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )

    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise RuntimeError("No bar data returned from Alpaca.")

    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(TIMEZONE)
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    bars = bars[
        (bars["timestamp"].dt.time >= time(9, 30))
        & (bars["timestamp"].dt.time <= time(16, 0))
    ].copy()

    frames = []
    for symbol in symbols:
        sdf = bars[bars["symbol"] == symbol].copy()
        sdf = sdf.tail(lookback_bars).reset_index(drop=True)
        if len(sdf) >= 40:
            frames.append(sdf)

    if not frames:
        raise RuntimeError("Not enough regular-hours data to compute regime features.")

    return pd.concat(frames, ignore_index=True)


# =========================================================
# FEATURE HELPERS
# =========================================================
def _safe_pct_change(series: pd.Series, periods: int) -> float:
    value = series.pct_change(periods).iloc[-1]
    return float(value) if pd.notna(value) else 0.0


def _safe_std(series: pd.Series, window: int) -> float:
    value = series.rolling(window).std().iloc[-1]
    return float(value) if pd.notna(value) else 0.0


def _safe_zscore(series: pd.Series, window: int) -> float:
    ma = series.rolling(window).mean().iloc[-1]
    std = series.rolling(window).std().iloc[-1]
    last = series.iloc[-1]
    if pd.isna(ma) or pd.isna(std) or std < 1e-12:
        return 0.0
    return float((last - ma) / std)


def _safe_slope(series: pd.Series, window: int) -> float:
    tail = series.tail(window)
    if len(tail) < window:
        return 0.0
    y = tail.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    denom = abs(y[-1]) + 1e-9
    return float(slope / denom)


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def compute_symbol_analysis(sdf: pd.DataFrame) -> SymbolAnalysis:
    sdf = sdf.copy()
    sdf["log_return"] = np.log(sdf["close"]).diff()

    close = sdf["close"]
    open_ = sdf["open"]
    high = sdf["high"]
    low = sdf["low"]

    ret_5 = _safe_pct_change(close, 5)
    ret_15 = _safe_pct_change(close, 15)
    ret_30 = _safe_pct_change(close, 30)
    vol_20 = _safe_std(sdf["log_return"], VOL_WINDOW)
    slope_15 = _safe_slope(close, TREND_WINDOW)
    zscore_20 = _safe_zscore(close, ZSCORE_WINDOW)

    last_close = float(close.iloc[-1])
    last_open = float(open_.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])

    prev_high_30 = float(high.shift(1).rolling(30).max().iloc[-1])
    prev_low_30 = float(low.shift(1).rolling(30).min().iloc[-1])

    breakout_attempted_prev_high = int(
        pd.notna(prev_high_30) and last_high >= prev_high_30 * (1.0 + FAILED_BREAKOUT_BUFFER)
    )
    failed_breakout_prev_high = int(
        breakout_attempted_prev_high
        and last_close < prev_high_30
        and last_close < last_open
    )

    breakdown_attempted_prev_low = int(
        pd.notna(prev_low_30) and last_low <= prev_low_30 * (1.0 - FAILED_BREAKOUT_BUFFER)
    )
    failed_breakdown_prev_low = int(
        breakdown_attempted_prev_low
        and last_close > prev_low_30
        and last_close > last_open
    )

    directional_score = float(
        0.40 * ret_5
        + 0.35 * ret_15
        + 0.25 * ret_30
        + 0.30 * slope_15
        + 0.10 * np.tanh(zscore_20 / 3.0)
    )

    pressure_score = float(
        0.50 * directional_score
        - 0.25 * failed_breakout_prev_high
        + 0.25 * failed_breakdown_prev_low
    )

    return SymbolAnalysis(
        symbol=str(sdf["symbol"].iloc[-1]),
        last_close=last_close,
        last_open=last_open,
        last_high=last_high,
        last_low=last_low,
        prev_high_30=prev_high_30 if np.isfinite(prev_high_30) else 0.0,
        prev_low_30=prev_low_30 if np.isfinite(prev_low_30) else 0.0,
        ret_5=ret_5,
        ret_15=ret_15,
        ret_30=ret_30,
        vol_20=vol_20,
        slope_15=slope_15,
        zscore_20=zscore_20,
        breakout_attempted_prev_high=breakout_attempted_prev_high,
        failed_breakout_prev_high=failed_breakout_prev_high,
        breakdown_attempted_prev_low=breakdown_attempted_prev_low,
        failed_breakdown_prev_low=failed_breakdown_prev_low,
        directional_score=directional_score,
        pressure_score=pressure_score,
    )


# =========================================================
# REGIME INFERENCE
# =========================================================
def infer_regime(analyses: Dict[str, SymbolAnalysis]) -> tuple[str, float, float, float, float]:
    spy = analyses["SPY"].directional_score
    dia = analyses["DIA"].directional_score
    qqq = analyses["QQQ"].directional_score
    tqqq = analyses["TQQQ"].directional_score
    iwm = analyses["IWM"].directional_score

    spxu = analyses["SPXU"].directional_score
    dog = analyses["DOG"].directional_score
    sqqq = analyses["SQQQ"].directional_score
    vixy = analyses["VIXY"].directional_score

    spy_fail_hi = analyses["SPY"].failed_breakout_prev_high
    qqq_fail_hi = analyses["QQQ"].failed_breakout_prev_high
    dia_fail_hi = analyses["DIA"].failed_breakout_prev_high

    spy_fail_lo = analyses["SPY"].failed_breakdown_prev_low
    qqq_fail_lo = analyses["QQQ"].failed_breakdown_prev_low
    dia_fail_lo = analyses["DIA"].failed_breakdown_prev_low

    bull_raw = np.mean(
        [
            spy,
            dia,
            qqq,
            tqqq,
            iwm,
            -spxu,
            -dog,
            -sqqq,
            -vixy,
        ]
    )

    bear_raw = np.mean(
        [
            -spy,
            -dia,
            -qqq,
            -tqqq,
            -iwm,
            spxu,
            dog,
            sqqq,
            vixy,
        ]
    )

    # failed breakouts in broad indexes hurt bull score
    bull_raw -= 0.10 * np.mean([spy_fail_hi, qqq_fail_hi, dia_fail_hi])

    # failed breakdowns in broad indexes hurt bear score
    bear_raw -= 0.10 * np.mean([spy_fail_lo, qqq_fail_lo, dia_fail_lo])

    bull_score = float(np.tanh(bull_raw * 12.0))
    bear_score = float(np.tanh(bear_raw * 12.0))

    score_gap = bull_score - bear_score
    abs_gap = abs(score_gap)
    transition_score = float(max(0.0, 1.0 - abs_gap))

    if bull_score >= BULL_THRESHOLD and score_gap >= TRANSITION_GAP_THRESHOLD:
        inferred_regime = "BULL"
        confidence = abs_gap
    elif bear_score >= BEAR_THRESHOLD and -score_gap >= TRANSITION_GAP_THRESHOLD:
        inferred_regime = "BEAR"
        confidence = abs_gap
    else:
        inferred_regime = "TRANSITION"
        confidence = 1.0 - min(abs_gap, 1.0)

    return inferred_regime, confidence, bull_score, bear_score, transition_score


# =========================================================
# REVERSAL MONITORING
# =========================================================
def detect_reversal_candidate(
    active_regime: str,
    inferred_regime: str,
    confidence: float,
    analyses: Dict[str, SymbolAnalysis],
) -> tuple[int, str]:
    spy_failed_high = analyses["SPY"].failed_breakout_prev_high
    qqq_failed_high = analyses["QQQ"].failed_breakout_prev_high
    dia_failed_high = analyses["DIA"].failed_breakout_prev_high

    spxu_strength = analyses["SPXU"].directional_score > 0
    sqqq_strength = analyses["SQQQ"].directional_score > 0
    dog_strength = analyses["DOG"].directional_score > 0
    vixy_strength = analyses["VIXY"].directional_score > 0

    reversal_watch = 0
    candidate_regime = ""

    if active_regime == "BULL":
        failed_breakout_pressure = any([spy_failed_high, qqq_failed_high, dia_failed_high])
        bear_confirmation = sum([spxu_strength, sqqq_strength, dog_strength, vixy_strength]) >= 2

        if failed_breakout_pressure:
            reversal_watch = 1
            candidate_regime = "TRANSITION"

            if inferred_regime == "BEAR" and confidence >= MIN_CONFIDENCE and bear_confirmation:
                candidate_regime = "BEAR"

    elif active_regime == "BEAR":
        spy_failed_low = analyses["SPY"].failed_breakdown_prev_low
        qqq_failed_low = analyses["QQQ"].failed_breakdown_prev_low
        dia_failed_low = analyses["DIA"].failed_breakdown_prev_low

        bull_confirmation = sum(
            [
                analyses["SPY"].directional_score > 0,
                analyses["QQQ"].directional_score > 0,
                analyses["DIA"].directional_score > 0,
                analyses["TQQQ"].directional_score > 0,
            ]
        ) >= 2

        if any([spy_failed_low, qqq_failed_low, dia_failed_low]):
            reversal_watch = 1
            candidate_regime = "TRANSITION"

            if inferred_regime == "BULL" and confidence >= MIN_CONFIDENCE and bull_confirmation:
                candidate_regime = "BULL"

    elif active_regime == "TRANSITION":
        if inferred_regime in {"BULL", "BEAR"} and confidence >= MIN_CONFIDENCE:
            reversal_watch = 1
            candidate_regime = inferred_regime

    return reversal_watch, candidate_regime


def update_active_regime(
    state: dict,
    inferred_regime: str,
    confidence: float,
    analyses: Dict[str, SymbolAnalysis],
) -> tuple[str, int, str, int, int]:
    old_active = state.get("active_regime", "TRANSITION")
    active_regime = old_active
    stored_candidate = state.get("candidate_regime", "")
    candidate_count = int(state.get("candidate_count", 0))

    reversal_watch, candidate_regime = detect_reversal_candidate(
        active_regime=active_regime,
        inferred_regime=inferred_regime,
        confidence=confidence,
        analyses=analyses,
    )

    flip_confirmed = 0

    if not reversal_watch or candidate_regime == "":
        state["candidate_regime"] = ""
        state["candidate_count"] = 0
        return active_regime, 0, "", 0, 0

    if candidate_regime == stored_candidate:
        candidate_count += 1
    else:
        candidate_count = 1

    state["candidate_regime"] = candidate_regime
    state["candidate_count"] = candidate_count

    if candidate_count >= REVERSAL_CONFIRM_COUNT:
        active_regime = candidate_regime
        state["active_regime"] = active_regime
        state["candidate_regime"] = ""
        state["candidate_count"] = 0
        flip_confirmed = int(active_regime != old_active)
        return active_regime, 0, "", 0, flip_confirmed

    return active_regime, 1, candidate_regime, candidate_count, 0


# =========================================================
# UNIVERSE SELECTION
# =========================================================
def select_universe(active_regime: str, confidence: float) -> tuple[str, int, List[str]]:
    if active_regime == "BEAR" and confidence >= MIN_CONFIDENCE:
        return "INVERSE_ETF", 1, INVERSE_ETF_UNIVERSE

    if active_regime == "BULL" and confidence >= MIN_CONFIDENCE:
        return "NORMAL", 1, NORMAL_UNIVERSE

    return "NONE", 0, []


# =========================================================
# CSV OUTPUT
# =========================================================
def flatten_for_csv(
    regime_result: RegimeResult,
    analyses: Dict[str, SymbolAnalysis],
) -> pd.DataFrame:
    row: Dict[str, float | str | int] = asdict(regime_result)

    for symbol, analysis in analyses.items():
        prefix = symbol.lower()
        row[f"{prefix}_close"] = analysis.last_close
        row[f"{prefix}_open"] = analysis.last_open
        row[f"{prefix}_high"] = analysis.last_high
        row[f"{prefix}_low"] = analysis.last_low
        row[f"{prefix}_prev_high_30"] = analysis.prev_high_30
        row[f"{prefix}_prev_low_30"] = analysis.prev_low_30
        row[f"{prefix}_ret_5"] = analysis.ret_5
        row[f"{prefix}_ret_15"] = analysis.ret_15
        row[f"{prefix}_ret_30"] = analysis.ret_30
        row[f"{prefix}_vol_20"] = analysis.vol_20
        row[f"{prefix}_slope_15"] = analysis.slope_15
        row[f"{prefix}_zscore_20"] = analysis.zscore_20
        row[f"{prefix}_breakout_attempted_prev_high"] = analysis.breakout_attempted_prev_high
        row[f"{prefix}_failed_breakout_prev_high"] = analysis.failed_breakout_prev_high
        row[f"{prefix}_breakdown_attempted_prev_low"] = analysis.breakdown_attempted_prev_low
        row[f"{prefix}_failed_breakdown_prev_low"] = analysis.failed_breakdown_prev_low
        row[f"{prefix}_directional_score"] = analysis.directional_score
        row[f"{prefix}_pressure_score"] = analysis.pressure_score

    return pd.DataFrame([row])


def append_csv(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_market_regime_classifier(
    output_csv: Path = OUTPUT_CSV,
    state_json: Path = STATE_JSON,
) -> tuple[RegimeResult, Dict[str, SymbolAnalysis], pd.DataFrame]:
    state = load_state(state_json)
    old_active = state.get("active_regime", "TRANSITION")

    client = get_client()
    bars = fetch_bars(client, MONITOR_SYMBOLS)

    analyses: Dict[str, SymbolAnalysis] = {}
    for symbol in MONITOR_SYMBOLS:
        sdf = bars[bars["symbol"] == symbol].copy()
        if len(sdf) == 0:
            continue
        analyses[symbol] = compute_symbol_analysis(sdf)

    missing = [s for s in MONITOR_SYMBOLS if s not in analyses]
    if missing:
        raise RuntimeError(f"Missing symbol analyses for: {missing}")

    inferred_regime, confidence, bull_score, bear_score, transition_score = infer_regime(analyses)

    active_regime, reversal_watch, candidate_regime, candidate_count, flip_confirmed = update_active_regime(
        state=state,
        inferred_regime=inferred_regime,
        confidence=confidence,
        analyses=analyses,
    )

    # extra safety if state was changed elsewhere
    flip_confirmed = int(active_regime != old_active) if flip_confirmed == 0 else flip_confirmed

    selected_universe, trading_enabled, symbols = select_universe(active_regime, confidence)

    regime_result = RegimeResult(
        timestamp=pd.Timestamp.now(tz=TIMEZONE).isoformat(),
        inferred_regime=inferred_regime,
        active_regime=active_regime,
        confidence=float(confidence),
        bull_score=float(bull_score),
        bear_score=float(bear_score),
        transition_score=float(transition_score),
        score_gap=float(bull_score - bear_score),
        reversal_watch=int(reversal_watch),
        candidate_regime=candidate_regime,
        candidate_count=int(candidate_count),
        flip_confirmed=int(flip_confirmed),
        selected_universe=selected_universe,
        trading_enabled=int(trading_enabled),
        universe_size=len(symbols),
        universe_symbols=",".join(symbols),
        regime_inferred=regime_to_num(inferred_regime),
        regime_active=regime_to_num(active_regime),
        selected_universe_num=universe_to_num(selected_universe),
    )

    save_state(state, state_json)

    out_df = flatten_for_csv(regime_result, analyses)
    append_csv(out_df, output_csv)

    return regime_result, analyses, out_df


def print_summary(regime_result: RegimeResult, analyses: Dict[str, SymbolAnalysis]) -> None:
    print("\n=== MARKET REGIME / UNIVERSE SELECTOR ===")
    print(f"timestamp:            {regime_result.timestamp}")
    print(f"inferred_regime:      {regime_result.inferred_regime}")
    print(f"active_regime:        {regime_result.active_regime}")
    print(f"confidence:           {regime_result.confidence:.4f}")
    print(f"bull_score:           {regime_result.bull_score:.4f}")
    print(f"bear_score:           {regime_result.bear_score:.4f}")
    print(f"transition_score:     {regime_result.transition_score:.4f}")
    print(f"score_gap:            {regime_result.score_gap:.4f}")
    print(f"reversal_watch:       {regime_result.reversal_watch}")
    print(f"candidate_regime:     {regime_result.candidate_regime}")
    print(f"candidate_count:      {regime_result.candidate_count}")
    print(f"flip_confirmed:       {regime_result.flip_confirmed}")
    print(f"selected_universe:    {regime_result.selected_universe}")
    print(f"trading_enabled:      {regime_result.trading_enabled}")
    print(f"universe_size:        {regime_result.universe_size}")
    print(f"regime_inferred_num:  {regime_result.regime_inferred}")
    print(f"regime_active_num:    {regime_result.regime_active}")
    print(f"universe_num:         {regime_result.selected_universe_num}")

    print("\n=== SYMBOL ANALYSIS ===")
    for symbol in MONITOR_SYMBOLS:
        a = analyses[symbol]
        print(
            f"{symbol:5s} "
            f"ret_5={a.ret_5:+.4f} "
            f"ret_15={a.ret_15:+.4f} "
            f"ret_30={a.ret_30:+.4f} "
            f"fail_hi={a.failed_breakout_prev_high} "
            f"fail_lo={a.failed_breakdown_prev_low} "
            f"dir={a.directional_score:+.4f} "
            f"pressure={a.pressure_score:+.4f}"
        )


if __name__ == "__main__":
    regime_result, analyses, _ = run_market_regime_classifier()
    print_summary(regime_result, analyses)
    print(f"\nSaved CSV row to: {OUTPUT_CSV}")
    print(f"Saved state to:   {STATE_JSON}")