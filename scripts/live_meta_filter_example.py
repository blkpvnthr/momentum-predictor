from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "meta_model.joblib"

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]


def build_feature_row(
    *,
    timestamp,
    signal: str,
    pred_return_5m: float,
    pred_return_15m: float,
    pred_return_30m: float,
    confidence: float,
) -> pd.DataFrame:
    ts = pd.Timestamp(timestamp)

    row = pd.DataFrame(
        [
            {
                "signal": signal,
                "pred_return_5m": pred_return_5m,
                "pred_return_15m": pred_return_15m,
                "pred_return_30m": pred_return_30m,
                "confidence": confidence,
                "confidence_abs_pred_15m": abs(pred_return_15m),
                "pred_sign_15m": np.sign(pred_return_15m),
                "pred_return_spread_5_15": pred_return_15m - pred_return_5m,
                "pred_return_spread_15_30": pred_return_30m - pred_return_15m,
                "hour": ts.hour,
                "minute": ts.minute,
                "minutes_from_open": (ts.hour * 60 + ts.minute) - (9 * 60 + 30),
            }
        ]
    )
    return row


def should_take_trade(
    *,
    timestamp,
    signal: str,
    pred_return_5m: float,
    pred_return_15m: float,
    pred_return_30m: float,
    confidence: float,
    threshold: float = 0.55,
) -> tuple[bool, float]:
    if signal != "LONG":
        return False, 0.0

    row = build_feature_row(
        timestamp=timestamp,
        signal=signal,
        pred_return_5m=pred_return_5m,
        pred_return_15m=pred_return_15m,
        pred_return_30m=pred_return_30m,
        confidence=confidence,
    )

    prob = float(pipeline.predict_proba(row)[0, 1])
    return prob >= threshold, prob


if __name__ == "__main__":
    take, prob = should_take_trade(
        timestamp="2026-04-05 10:15:00-04:00",
        signal="LONG",
        pred_return_5m=0.0012,
        pred_return_15m=0.0028,
        pred_return_30m=0.0034,
        confidence=0.81,
        threshold=0.55,
    )
    print("take_trade:", take)
    print("meta_prob:", prob)