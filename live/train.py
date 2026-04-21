from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trading_env import TradeCentricMDPConfig, TradingEnv
from trading_system import REGIME_ENGINE_PATH, build_features


THIS_FILE = Path(__file__).resolve()
LIVE_DIR = THIS_FILE.parent
PROJECT_ROOT = LIVE_DIR.parent

ENV_PATH = PROJECT_ROOT / ".env"
TRAINING_DATA_PATH = LIVE_DIR / "training_data.csv"
TENSORBOARD_DIR = LIVE_DIR / "logs" / "tensorboard"
CACHE_DIR = LIVE_DIR / "download_cache"

ENSEMBLE_DIR = LIVE_DIR / "ensemble_models"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR = ENSEMBLE_DIR / "validation_curves"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = ENSEMBLE_DIR / "window_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CORE_SYMBOLS = ["QQQ", "TQQQ", "SQQQ"]
UNIVERSE_SYMBOLS = [
    "SPY", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV", "SOXX", "SMH", "ARKK", "VIXY",
    "TQQQ", "SQQQ", "SOXL", "SOXS", "TECL", "SPXL", "SPXU", "DOG", "DXD", "SRTY", "SDOW",
    "AMD", "IONQ", "QBTS", "RGTI", "QUBT",
    "ASTS", "LUNR",
]
SYMBOLS = sorted(set(CORE_SYMBOLS + UNIVERSE_SYMBOLS))

START = "2024-01-01"
END = "2026-04-01"
TIMEZONE = "America/New_York"

SYMBOL_CHUNK_SIZE = 15
DATE_CHUNK_DAYS = 5
REQUEST_PAUSE_SECONDS = 1.5
MAX_RETRIES = 6
BACKOFF_BASE_SECONDS = 3.0
MAX_DOWNLOAD_WORKERS = 4

TRAIN_MONTHS_INITIAL = 6
RETRAIN_EVERY_MONTHS = 3
VALIDATION_MONTHS = 3
TRADE_MONTHS = 3

TIMESTEPS_PPO = 250_000
TIMESTEPS_A2C = 250_000
TIMESTEPS_DDPG = 250_000
RISK_FREE_RATE_ANNUAL = 0.02


def load_alpaca_client() -> StockHistoricalDataClient:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    print(f"[ensemble] loading env from: {ENV_PATH}")
    print(f"[ensemble] APCA_API_KEY_ID present: {bool(api_key)}")
    print(f"[ensemble] APCA_API_SECRET_KEY present: {bool(secret_key)}")

    if not api_key or not secret_key:
        raise RuntimeError("Missing Alpaca credentials in .env")

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def _normalize_bar_frame(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or len(bars) == 0:
        return pd.DataFrame()

    bars = bars.copy()

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index()
    elif isinstance(bars.index, pd.DatetimeIndex):
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
    if "trade_count" not in bars.columns and "n" in bars.columns:
        rename_map["n"] = "trade_count"
    if "vwap" not in bars.columns and "vw" in bars.columns:
        rename_map["vw"] = "vwap"

    if rename_map:
        bars = bars.rename(columns=rename_map)

    required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise RuntimeError(f"Downloaded bars missing required columns: {missing}")

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["timestamp"])

    if len(bars) == 0:
        return pd.DataFrame()

    bars["timestamp"] = bars["timestamp"].dt.tz_convert(TIMEZONE)

    bars = (
        bars.sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    bars = bars[
        (bars["timestamp"].dt.time >= dt_time(9, 30))
        & (bars["timestamp"].dt.time <= dt_time(16, 0))
    ].copy()

    if len(bars) == 0:
        return pd.DataFrame()

    return bars


def _symbol_chunks(symbols: list[str], chunk_size: int) -> list[list[str]]:
    return [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]


def _date_chunks(start: str, end: str, chunk_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end_ts)
        out.append((cur, nxt))
        cur = nxt
    return out


def _cache_file(symbols: list[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sym_tag = "-".join(symbols)
    fname = f"{sym_tag}__{start_ts.strftime('%Y%m%d')}__{end_ts.strftime('%Y%m%d')}.csv"
    return CACHE_DIR / fname


def _download_request_with_backoff(client, symbols, start_ts, end_ts) -> pd.DataFrame:
    cache_path = _cache_file(symbols, start_ts, end_ts)
    if cache_path.exists():
        print(f"[ensemble] cache hit | {cache_path.name}")
        cached = pd.read_csv(cache_path)
        return _normalize_bar_frame(cached)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start_ts,
        end=end_ts,
    )

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"[ensemble] download attempt {attempt}/{MAX_RETRIES} | "
                f"symbols={symbols} | start={start_ts.date()} | end={end_ts.date()}"
            )
            bars = client.get_stock_bars(request).df
            bars = _normalize_bar_frame(bars)
            if bars is None or len(bars) == 0:
                raise RuntimeError("No regular-hours bars returned")
            bars.to_csv(cache_path, index=False)
            time.sleep(REQUEST_PAUSE_SECONDS)
            return bars
        except (APIError, Exception) as e:
            last_err = e
            msg = str(e)
            is_rate_limit = ("429" in msg) or ("Too Many Requests" in msg)
            if attempt == MAX_RETRIES:
                break
            sleep_s = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            if is_rate_limit:
                sleep_s *= 1.5
            print(f"[ensemble] request failed | sleeping {sleep_s:.1f}s | error={msg}")
            time.sleep(sleep_s)

    raise RuntimeError("Failed to download Alpaca bars after retries") from last_err


def _download_one_job(client, sym_chunk, dstart, dend, job_idx, total_jobs) -> pd.DataFrame:
    print(f"[ensemble] chunk job {job_idx}/{total_jobs} | date={dstart.date()}->{dend.date()} | symbols={sym_chunk}")
    return _download_request_with_backoff(client, sym_chunk, dstart, dend)


def _download_all_chunks(client, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    date_chunks = _date_chunks(start, end, DATE_CHUNK_DAYS)
    symbol_chunks = _symbol_chunks(symbols, SYMBOL_CHUNK_SIZE)

    jobs = []
    total_jobs = len(date_chunks) * len(symbol_chunks)
    job_idx = 0

    for dstart, dend in date_chunks:
        for sym_chunk in symbol_chunks:
            job_idx += 1
            jobs.append((sym_chunk, dstart, dend, job_idx, total_jobs))

    parts: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as ex:
        futures = [
            ex.submit(_download_one_job, client, sym_chunk, dstart, dend, idx, total)
            for sym_chunk, dstart, dend, idx, total in jobs
        ]
        for fut in as_completed(futures):
            part = fut.result()
            if part is not None and not part.empty:
                parts.append(part)

    if not parts:
        raise RuntimeError("No chunked bar data was downloaded.")

    bars = pd.concat(parts, ignore_index=True)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce", utc=True)
    bars = bars.dropna(subset=["timestamp"])

    bars = (
        bars.sort_values(["symbol", "timestamp"])
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .reset_index(drop=True)
    )

    if len(bars) == 0:
        raise RuntimeError("Downloaded bars dataframe is empty after concat/dedup.")

    return bars


def _pivot_long_bars_to_wide(bars: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()

    df = bars.copy()

    required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"_pivot_long_bars_to_wide: missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])

    # convert to naive timestamps for downstream consistency
    df["timestamp"] = df["timestamp"].dt.tz_convert(TIMEZONE).dt.tz_localize(None)

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df[df["symbol"].isin([s.upper() for s in symbols])].copy()

    if df.empty:
        return pd.DataFrame()

    value_cols = [c for c in ["open", "high", "low", "close", "volume", "trade_count", "vwap"] if c in df.columns]

    wide_parts = []
    for col in value_cols:
        pivoted = (
            df.pivot_table(
                index="timestamp",
                columns="symbol",
                values=col,
                aggfunc="last",
            )
            .sort_index()
        )
        pivoted.columns = [f"{sym.lower()}_{col}" for sym in pivoted.columns]
        wide_parts.append(pivoted)

    if not wide_parts:
        return pd.DataFrame()

    merged = pd.concat(wide_parts, axis=1).reset_index()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return merged


def prepare_feature_frame(raw: pd.DataFrame, base_symbol: str = "QQQ") -> pd.DataFrame:
    if raw is None or raw.empty or len(raw.columns) == 0:
        raise ValueError("prepare_feature_frame: raw dataframe is empty")

    df = raw.copy()
    prefix = base_symbol.lower()

    required_map = {
        "timestamp": "timestamp",
        f"{prefix}_open": "open",
        f"{prefix}_high": "high",
        f"{prefix}_low": "low",
        f"{prefix}_close": "close",
        f"{prefix}_volume": "volume",
        "tqqq_close": "tqqq_close",
        "sqqq_close": "sqqq_close",
    }

    missing = [src for src in required_map if src not in df.columns]
    if missing:
        raise ValueError(
            f"prepare_feature_frame: missing required source columns: {missing}\n"
            f"available columns sample: {list(df.columns)[:50]}"
        )

    for src, dst in required_map.items():
        if src != dst:
            df[dst] = df[src]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def download_training_data(
    symbols: list[str] = SYMBOLS,
    start: str = START,
    end: str = END,
    output_path: Path = TRAINING_DATA_PATH,
) -> pd.DataFrame:
    client = load_alpaca_client()
    print(f"[ensemble] requesting Alpaca bars | symbol_count={len(symbols)} | start={start} | end={end}")

    bars = _download_all_chunks(client=client, symbols=symbols, start=start, end=end)
    print(f"[ensemble] downloaded long bars rows={len(bars):,}")

    merged = _pivot_long_bars_to_wide(bars, symbols)
    if merged.empty:
        raise RuntimeError("download_training_data: wide merged dataframe is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"[ensemble] saved {len(merged):,} merged rows to {output_path}")
    print(f"[ensemble] merged columns sample: {list(merged.columns)[:20]}")

    return merged


def make_env(df: pd.DataFrame):
    cfg = TradeCentricMDPConfig()

    def _init():
        return TradingEnv(data=df, config=cfg)

    return _init


def annualized_sharpe(
    equity_curve: pd.Series,
    risk_free_rate_annual: float = RISK_FREE_RATE_ANNUAL,
) -> float:
    returns = equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        return -np.inf
    periods_per_year = 252 * 390
    rf_per_period = risk_free_rate_annual / periods_per_year
    excess = returns - rf_per_period
    std = excess.std()
    if std <= 1e-12:
        return -np.inf
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def _build_model(agent_name: str, env):
    if agent_name == "ppo":
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=7.5e-5,
            n_steps=2048,
            batch_size=128,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0025,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(TENSORBOARD_DIR / "ppo"),
        )
    if agent_name == "a2c":
        return A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=7.5e-5,
            n_steps=64,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.001,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(TENSORBOARD_DIR / "a2c"),
        )
    if agent_name == "ddpg":
        return DDPG(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=200_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            train_freq=(1, "step"),
            gradient_steps=1,
            verbose=1,
            tensorboard_log=str(TENSORBOARD_DIR / "ddpg"),
        )
    raise ValueError(agent_name)


def timesteps_for_agent(name: str) -> int:
    return {"ppo": TIMESTEPS_PPO, "a2c": TIMESTEPS_A2C, "ddpg": TIMESTEPS_DDPG}[name]


def rollout_equity_curve(model, vecnorm: VecNormalize, df: pd.DataFrame, label: str) -> pd.DataFrame:
    env = TradingEnv(df.copy(), config=TradeCentricMDPConfig(max_episode_steps=max(len(df) - 1, 1)))
    obs, reset_info = env.reset()
    records = []

    done = False
    truncated = False
    while not (done or truncated):
        obs_in = vecnorm.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(obs_in, deterministic=True)
        action = np.asarray(action).reshape(-1)
        obs, reward, done, truncated, info = env.step(action)
        records.append(
            {
                "label": label,
                "timestamp": df.iloc[env.idx]["timestamp"] if "timestamp" in df.columns else env.idx,
                "portfolio_value": float(info["portfolio_value"]),
                "reward": float(reward),
                "turbulence": float(info["turbulence"]),
                "turbulence_threshold": float(info["turbulence_threshold"]),
                "turbulence_triggered": bool(float(info["turbulence"]) > float(info["turbulence_threshold"])),
                "balance": float(info["balance"]),
            }
        )
    return pd.DataFrame(records)


def summarize_turbulence(curve: pd.DataFrame) -> dict:
    if len(curve) == 0:
        return {
            "turbulence_event_count": 0,
            "turbulence_event_ratio": 0.0,
            "max_turbulence": np.nan,
            "threshold": np.nan,
        }
    return {
        "turbulence_event_count": int(curve["turbulence_triggered"].sum()),
        "turbulence_event_ratio": float(curve["turbulence_triggered"].mean()),
        "max_turbulence": float(curve["turbulence"].max()),
        "threshold": float(curve["turbulence_threshold"].iloc[0]),
    }


def month_windows(df: pd.DataFrame) -> list[pd.Timestamp]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")

    ts = ts.dt.tz_localize(None)

    months = ts.dt.to_period("M").drop_duplicates().sort_values()
    return [m.to_timestamp() for m in months]


def slice_by_months(df: pd.DataFrame, start_month: pd.Timestamp, end_month_exclusive: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")

    ts = ts.dt.tz_localize(None)

    start_month = pd.Timestamp(start_month)
    end_month_exclusive = pd.Timestamp(end_month_exclusive)

    if start_month.tzinfo is not None:
        start_month = start_month.tz_localize(None)
    if end_month_exclusive.tzinfo is not None:
        end_month_exclusive = end_month_exclusive.tz_localize(None)

    mask = (ts >= start_month) & (ts < end_month_exclusive)
    return df.loc[mask].copy().reset_index(drop=True)


def add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    return (ts.to_period("M") + months).to_timestamp()


def checkpoint_path_for_window(run_name: str) -> Path:
    return CHECKPOINT_DIR / f"{run_name}.json"


def load_window_checkpoint(run_name: str) -> dict:
    path = checkpoint_path_for_window(run_name)
    if path.exists():
        return json.loads(path.read_text())
    return {"run_name": run_name, "agents": {}, "selected_best": None, "trade_done": False}


def save_window_checkpoint(run_name: str, payload: dict) -> None:
    path = checkpoint_path_for_window(run_name)
    path.write_text(json.dumps(payload, indent=2, default=str))


def train_and_validate_agent(agent_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, run_name: str):
    train_env = DummyVecEnv([make_env(train_df)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    model = _build_model(agent_name, train_env)
    print(f"[ensemble] training {agent_name} | run={run_name}")
    model.learn(total_timesteps=timesteps_for_agent(agent_name))

    val_curve = rollout_equity_curve(model, train_env, val_df, label=f"{run_name}_{agent_name}_validation")
    sharpe = annualized_sharpe(val_curve["portfolio_value"])
    turb = summarize_turbulence(val_curve)

    model_path = ENSEMBLE_DIR / f"{run_name}_{agent_name}.zip"
    vecnorm_path = ENSEMBLE_DIR / f"{run_name}_{agent_name}_vecnorm.pkl"
    val_curve_path = VALIDATION_DIR / f"{run_name}_{agent_name}_validation_curve.csv"

    model.save(str(model_path))
    train_env.save(str(vecnorm_path))
    val_curve.to_csv(val_curve_path, index=False)

    return {
        "agent": agent_name,
        "sharpe": sharpe,
        "model_path": str(model_path),
        "vecnorm_path": str(vecnorm_path),
        "validation_curve_path": str(val_curve_path),
        "validation_rows": int(len(val_curve)),
        "final_portfolio_value": float(val_curve["portfolio_value"].iloc[-1]) if len(val_curve) else np.nan,
        **turb,
    }


def load_model_and_vecnorm(agent_name: str, model_path: str, vecnorm_path: str, ref_df: pd.DataFrame):
    ref_env = DummyVecEnv([make_env(ref_df)])
    vecnorm = VecNormalize.load(str(vecnorm_path), ref_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    if agent_name == "ppo":
        model = PPO.load(str(model_path))
    elif agent_name == "a2c":
        model = A2C.load(str(model_path))
    elif agent_name == "ddpg":
        model = DDPG.load(str(model_path))
    else:
        raise ValueError(agent_name)

    return model, vecnorm


def main() -> None:
    print("[ensemble] starting trade-centric walk-forward ensemble training pipeline...")
    raw = download_training_data(symbols=SYMBOLS, start=START, end=END, output_path=TRAINING_DATA_PATH)

    print(f"[ensemble] raw rows={len(raw):,}")
    print(f"[ensemble] raw columns sample={list(raw.columns)[:20]}")
    print("[ensemble] building features and fitting regime engine...")

    raw = prepare_feature_frame(raw, base_symbol="QQQ")
    df, regime_engine = build_features(raw, regime_engine=None, fit_regime_engine=True)

    if regime_engine is None:
        raise RuntimeError("Failed to fit regime engine.")
    if len(df) == 0:
        raise RuntimeError("No rows remained after feature engineering.")

    REGIME_ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    regime_engine.save(REGIME_ENGINE_PATH)

    months = month_windows(df)
    if len(months) < (TRAIN_MONTHS_INITIAL + VALIDATION_MONTHS + TRADE_MONTHS):
        raise RuntimeError("Not enough monthly history for rolling ensemble procedure.")

    results = []
    window_idx = 0
    start_month = months[0]

    while True:
        train_start = start_month
        train_end = add_months(train_start, TRAIN_MONTHS_INITIAL + window_idx * RETRAIN_EVERY_MONTHS)
        val_end = add_months(train_end, VALIDATION_MONTHS)
        trade_end = add_months(val_end, TRADE_MONTHS)

        if trade_end > months[-1] + pd.offsets.MonthBegin(1):
            break

        train_df = slice_by_months(df, train_start, train_end)
        val_df = slice_by_months(df, train_end, val_end)
        trade_df = slice_by_months(df, val_end, trade_end)

        if len(train_df) == 0 or len(val_df) == 0 or len(trade_df) == 0:
            break

        run_name = f"window_{window_idx}_{train_start.strftime('%Y%m')}_{trade_end.strftime('%Y%m')}"
        print(
            f"[ensemble] window={window_idx} | "
            f"train={train_start.date()}->{train_end.date()} | "
            f"val={train_end.date()}->{val_end.date()} | "
            f"trade={val_end.date()}->{trade_end.date()}"
        )

        ckpt = load_window_checkpoint(run_name)

        for agent_name in ("ppo", "a2c", "ddpg"):
            if agent_name in ckpt["agents"]:
                print(f"[ensemble] checkpoint hit | skipping agent={agent_name} | run={run_name}")
                continue

            result = train_and_validate_agent(agent_name, train_df, val_df, run_name)
            ckpt["agents"][agent_name] = result
            save_window_checkpoint(run_name, ckpt)

        window_results = list(ckpt["agents"].values())
        best = max(window_results, key=lambda x: x["sharpe"])
        ckpt["selected_best"] = best["agent"]
        save_window_checkpoint(run_name, ckpt)

        print(
            f"[ensemble] selected best agent={best['agent']} | "
            f"validation_sharpe={best['sharpe']:.4f} | "
            f"validation_final_value={best['final_portfolio_value']:.2f}"
        )

        if not ckpt.get("trade_done", False):
            model, vecnorm = load_model_and_vecnorm(
                agent_name=best["agent"],
                model_path=best["model_path"],
                vecnorm_path=best["vecnorm_path"],
                ref_df=train_df,
            )
            trade_curve = rollout_equity_curve(model, vecnorm, trade_df, label=f"{run_name}_{best['agent']}_trade")
            trade_curve_path = VALIDATION_DIR / f"{run_name}_{best['agent']}_trade_curve.csv"
            trade_curve.to_csv(trade_curve_path, index=False)
            trade_sharpe = annualized_sharpe(trade_curve["portfolio_value"]) if len(trade_curve) else -np.inf
            trade_turb = summarize_turbulence(trade_curve)

            ckpt["trade_done"] = True
            ckpt["trade_curve_path"] = str(trade_curve_path)
            ckpt["trade_sharpe"] = trade_sharpe
            ckpt["trade_final_value"] = float(trade_curve["portfolio_value"].iloc[-1]) if len(trade_curve) else np.nan
            ckpt["trade_turbulence"] = trade_turb
            save_window_checkpoint(run_name, ckpt)
        else:
            trade_sharpe = ckpt.get("trade_sharpe", -np.inf)
            trade_turb = ckpt.get("trade_turbulence", {})

        results.append(
            {
                "window": window_idx,
                "train_start": train_start,
                "train_end": train_end,
                "val_end": val_end,
                "trade_end": trade_end,
                "best_agent": best["agent"],
                "validation_sharpe": best["sharpe"],
                "validation_final_value": best["final_portfolio_value"],
                "validation_turbulence_events": best["turbulence_event_count"],
                "validation_turbulence_ratio": best["turbulence_event_ratio"],
                "trade_sharpe": trade_sharpe,
                "trade_final_value": ckpt.get("trade_final_value", np.nan),
                "trade_turbulence_events": trade_turb.get("turbulence_event_count", np.nan),
                "trade_turbulence_ratio": trade_turb.get("turbulence_event_ratio", np.nan),
                "checkpoint_path": str(checkpoint_path_for_window(run_name)),
            }
        )

        window_idx += 1

    results_df = pd.DataFrame(results)
    out_csv = ENSEMBLE_DIR / "ensemble_summary.csv"
    results_df.to_csv(out_csv, index=False)

    print("[ensemble] finished.")
    print(results_df.to_string(index=False))
    print(f"[ensemble] saved summary to {out_csv}")


if __name__ == "__main__":
    main()