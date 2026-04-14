from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from momentum_predictor.model import DualStreamTransformer


# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class PredictionRecord:
    timestamp: str
    pred_return_5m: float
    pred_return_15m: float
    pred_return_30m: float
    breakout_up_prob_15m: float
    breakout_down_prob_15m: float
    continuation_prob_15m: float
    confidence: float
    signal: str


# =========================================================
# DEVICE
# =========================================================
def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# =========================================================
# MODEL LOADING
# =========================================================
def build_model(
    input_dim_1m: int,
    input_dim_5m: int,
    device: str = "auto",
) -> tuple[DualStreamTransformer, torch.device]:
    device_obj = get_device(device)
    model = DualStreamTransformer(
        input_dim_1m=input_dim_1m,
        input_dim_5m=input_dim_5m,
    ).to(device_obj)
    model.eval()
    return model, device_obj


def load_model_checkpoint(
    checkpoint_path: str | Path,
    input_dim_1m: int | None = None,
    input_dim_5m: int | None = None,
    device: str = "auto",
) -> tuple[DualStreamTransformer, torch.device]:
    device_obj = get_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a dict-based checkpoint.")

    metadata = checkpoint.get("metadata", {})
    ckpt_input_dim_1m = metadata.get("input_dim_1m", input_dim_1m)
    ckpt_input_dim_5m = metadata.get("input_dim_5m", input_dim_5m)

    if ckpt_input_dim_1m is None or ckpt_input_dim_5m is None:
        raise RuntimeError("Checkpoint is missing input dimensions and none were provided.")

    model, _ = build_model(
        input_dim_1m=int(ckpt_input_dim_1m),
        input_dim_5m=int(ckpt_input_dim_5m),
        device=device,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint missing 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device_obj


# =========================================================
# MATH HELPERS
# =========================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_clip(x: float | np.ndarray, limit: float = 0.10) -> float:
    return float(np.clip(x, -limit, limit))


def compute_confidence(
    pred_return_15m: float,
    breakout_up_prob_15m: float,
    breakout_down_prob_15m: float,
    continuation_prob_15m: float,
) -> float:
    magnitude = abs(pred_return_15m)

    # broader scaling so larger outputs are not immediately saturated
    mag_score = min(magnitude / 0.01, 1.0)

    breakout_prob = max(breakout_up_prob_15m, breakout_down_prob_15m)
    structure_score = max(breakout_prob, continuation_prob_15m)

    return float(np.clip(0.55 * mag_score + 0.45 * structure_score, 0.0, 1.0))


def derive_signal_label(
    pred_return_15m: float,
    breakout_up_prob_15m: float,
    breakout_down_prob_15m: float,
    continuation_prob_15m: float,
    confidence: float,
    min_confidence: float = 0.45,
    min_breakout_prob: float = 0.30,
    min_return_abs: float = 0.0010,
) -> str:
    if confidence < min_confidence:
        return "NO_TRADE"

    if abs(pred_return_15m) < min_return_abs:
        return "NO_TRADE"

    if breakout_up_prob_15m >= min_breakout_prob and pred_return_15m > 0:
        if continuation_prob_15m >= 0.50:
            return "LONG_BREAKOUT_CONTINUATION"
        return "LONG_BREAKOUT_REVERSAL_RISK"

    if breakout_down_prob_15m >= min_breakout_prob and pred_return_15m < 0:
        if continuation_prob_15m >= 0.50:
            return "SHORT_BREAKOUT_CONTINUATION"
        return "SHORT_BREAKOUT_REVERSAL_RISK"

    if pred_return_15m > 0:
        return "LONG_BIAS"

    if pred_return_15m < 0:
        return "SHORT_BIAS"

    return "NO_TRADE"


# =========================================================
# SIGNAL ENGINES
# =========================================================
def generate_trading_signals(
    records: list[PredictionRecord],
    top_pct: float = 0.10,
    min_confidence: float = 0.45,
    min_pred_return: float = 0.0010,
) -> list[PredictionRecord]:
    if not records:
        return records

    returns = np.array([r.pred_return_15m for r in records])

    top_threshold = np.quantile(returns, 1 - top_pct)
    bottom_threshold = np.quantile(returns, top_pct)

    for r in records:
        if r.confidence < min_confidence:
            r.signal = "NO_TRADE"
            continue

        if r.pred_return_15m >= max(top_threshold, min_pred_return):
            r.signal = "LONG" if r.breakout_up_prob_15m >= r.breakout_down_prob_15m else "NO_TRADE"
        elif r.pred_return_15m <= min(bottom_threshold, -min_pred_return):
            r.signal = "SHORT" if r.breakout_down_prob_15m >= r.breakout_up_prob_15m else "NO_TRADE"
        else:
            r.signal = "NO_TRADE"

    return records


def apply_regime_filter(records: list[PredictionRecord], regime_series: Sequence[int]) -> list[PredictionRecord]:
    """
    regime_series must align with timestamps:
    +1 = bull
    -1 = bear
     0 = transition
    """
    for r, regime in zip(records, regime_series):
        if regime == 0:
            r.signal = "NO_TRADE"
        elif regime == 1 and r.signal == "SHORT":
            r.signal = "NO_TRADE"
        elif regime == -1 and r.signal == "LONG":
            r.signal = "NO_TRADE"

    return records


def apply_signal_engine(
    records: list[PredictionRecord],
    top_q: float = 0.85,
    bot_q: float = 0.15,
    min_confidence: float = 0.45,
    min_pred_return: float = 0.0010,
) -> list[PredictionRecord]:
    if not records:
        return records

    returns = np.array([r.pred_return_15m for r in records])
    top_thr = np.quantile(returns, top_q)
    bot_thr = np.quantile(returns, bot_q)

    for r in records:
        if r.confidence < min_confidence:
            r.signal = "NO_TRADE"
            continue

        if r.pred_return_15m >= max(top_thr, min_pred_return):
            r.signal = "LONG"
        elif r.pred_return_15m <= min(bot_thr, -min_pred_return):
            r.signal = "SHORT"
        else:
            r.signal = "NO_TRADE"

    return records


def apply_execution_filters(
    records: list[PredictionRecord],
    min_confidence: float = 0.45,
    min_pred_return: float = 0.0010,
    min_breakout_prob: float = 0.35,
    min_continuation_prob: float = 0.50,
) -> list[PredictionRecord]:
    for r in records:
        if r.confidence < min_confidence:
            r.signal = "NO_TRADE"
            continue

        if r.pred_return_15m >= min_pred_return:
            if (
                r.breakout_up_prob_15m >= min_breakout_prob
                or r.continuation_prob_15m >= min_continuation_prob
            ):
                r.signal = "LONG"
            else:
                r.signal = "NO_TRADE"

        elif r.pred_return_15m <= -min_pred_return:
            if (
                r.breakout_down_prob_15m >= min_breakout_prob
                or r.continuation_prob_15m >= min_continuation_prob
            ):
                r.signal = "SHORT"
            else:
                r.signal = "NO_TRADE"

        else:
            r.signal = "NO_TRADE"

    return records


# =========================================================
# CORE INFERENCE
# =========================================================
@torch.no_grad()
def predict_sequences(
    model: torch.nn.Module,
    X1: np.ndarray,
    X5: np.ndarray,
    timestamps: Sequence[str] | Sequence[np.datetime64] | Sequence,
    device: str | torch.device = "auto",
    batch_size: int = 512,
    center_predictions: bool = True,
    clip_predictions: bool = False,
    clip_limit: float = 0.10,
) -> list[PredictionRecord]:
    if len(X1) != len(X5):
        raise ValueError(f"X1 and X5 length mismatch: {len(X1)} vs {len(X5)}")

    if len(X1) != len(timestamps):
        raise ValueError(
            f"timestamps length mismatch: len(X1)={len(X1)} vs len(timestamps)={len(timestamps)}"
        )

    device_obj = get_device(device) if isinstance(device, str) else device
    model = model.to(device_obj)
    model.eval()

    pred_return_rows: list[np.ndarray] = []
    breakout_rows: list[np.ndarray] = []
    continuation_rows: list[np.ndarray] = []

    for start_idx in range(0, len(X1), batch_size):
        end_idx = min(start_idx + batch_size, len(X1))

        xb1 = torch.tensor(X1[start_idx:end_idx], dtype=torch.float32, device=device_obj)
        xb5 = torch.tensor(X5[start_idx:end_idx], dtype=torch.float32, device=device_obj)

        outputs = model(xb1, xb5)

        pred_returns = outputs["returns"].detach().cpu().numpy()
        breakout_logits = outputs["breakout"].detach().cpu().numpy()
        continuation_logits = outputs["continuation"].detach().cpu().numpy().reshape(-1)

        pred_return_rows.append(pred_returns)
        breakout_rows.append(breakout_logits)
        continuation_rows.append(continuation_logits)

    pred_returns = np.vstack(pred_return_rows)
    breakout_logits = np.vstack(breakout_rows)
    continuation_logits = np.concatenate(continuation_rows)

    if center_predictions:
        pred_returns = pred_returns - pred_returns.mean(axis=0, keepdims=True)

    if clip_predictions:
        pred_returns = np.clip(pred_returns, -clip_limit, clip_limit)

    print("[inference] raw prediction stats:")
    for idx, name in enumerate(["5m", "15m", "30m"]):
        col = pred_returns[:, idx]
        print(
            f"  {name}: mean={col.mean():.6f} std={col.std():.6f} "
            f"min={col.min():.6f} max={col.max():.6f}"
        )

    breakout_probs = sigmoid(breakout_logits)
    continuation_probs = sigmoid(continuation_logits)

    records: list[PredictionRecord] = []

    for i in range(len(X1)):
        pred_return_5m = float(pred_returns[i, 0])
        pred_return_15m = float(pred_returns[i, 1])
        pred_return_30m = float(pred_returns[i, 2])

        breakout_up_prob_15m = float(breakout_probs[i, 0])
        breakout_down_prob_15m = float(breakout_probs[i, 1])
        continuation_prob_15m = float(continuation_probs[i])

        confidence = compute_confidence(
            pred_return_15m=pred_return_15m,
            breakout_up_prob_15m=breakout_up_prob_15m,
            breakout_down_prob_15m=breakout_down_prob_15m,
            continuation_prob_15m=continuation_prob_15m,
        )

        signal = derive_signal_label(
            pred_return_15m=pred_return_15m,
            breakout_up_prob_15m=breakout_up_prob_15m,
            breakout_down_prob_15m=breakout_down_prob_15m,
            continuation_prob_15m=continuation_prob_15m,
            confidence=confidence,
        )

        records.append(
            PredictionRecord(
                timestamp=str(timestamps[i]),
                pred_return_5m=pred_return_5m,
                pred_return_15m=pred_return_15m,
                pred_return_30m=pred_return_30m,
                breakout_up_prob_15m=breakout_up_prob_15m,
                breakout_down_prob_15m=breakout_down_prob_15m,
                continuation_prob_15m=continuation_prob_15m,
                confidence=confidence,
                signal=signal,
            )
        )

    return records


# =========================================================
# CSV EXPORT
# =========================================================
def write_predictions_to_csv(
    records: Sequence[PredictionRecord],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "pred_return_5m",
        "pred_return_15m",
        "pred_return_30m",
        "breakout_up_prob_15m",
        "breakout_down_prob_15m",
        "continuation_prob_15m",
        "confidence",
        "signal",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            writer.writerow(
                {
                    "timestamp": r.timestamp,
                    "pred_return_5m": r.pred_return_5m,
                    "pred_return_15m": r.pred_return_15m,
                    "pred_return_30m": r.pred_return_30m,
                    "breakout_up_prob_15m": r.breakout_up_prob_15m,
                    "breakout_down_prob_15m": r.breakout_down_prob_15m,
                    "continuation_prob_15m": r.continuation_prob_15m,
                    "confidence": r.confidence,
                    "signal": r.signal,
                }
            )

    return output_path


# =========================================================
# CONVENIENCE WRAPPER
# =========================================================
def run_inference_to_csv(
    checkpoint_path: str | Path,
    X1: np.ndarray,
    X5: np.ndarray,
    timestamps: Sequence[str] | Sequence[np.datetime64] | Sequence,
    output_csv_path: str | Path,
    device: str = "auto",
    batch_size: int = 512,
    center_predictions: bool = True,
    use_execution_filters: bool = True,
    clip_predictions: bool = False,
    clip_limit: float = 0.10,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_csv_path = Path(output_csv_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if X1.ndim != 3 or X5.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays for X1 and X5, got X1.ndim={X1.ndim}, X5.ndim={X5.ndim}"
        )

    if len(X1) == 0 or len(X5) == 0:
        raise ValueError("X1/X5 are empty, so there is nothing to write to CSV.")

    if len(X1) != len(timestamps):
        raise ValueError(
            f"timestamps length mismatch: len(X1)={len(X1)} vs len(timestamps)={len(timestamps)}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a dict-based checkpoint.")

    metadata = checkpoint.get("metadata", {})
    expected_1m = metadata.get("input_dim_1m")
    expected_5m = metadata.get("input_dim_5m")

    if expected_1m is not None and X1.shape[-1] != expected_1m:
        raise ValueError(f"X1 feature mismatch: checkpoint expects {expected_1m}, got {X1.shape[-1]}")

    if expected_5m is not None and X5.shape[-1] != expected_5m:
        raise ValueError(f"X5 feature mismatch: checkpoint expects {expected_5m}, got {X5.shape[-1]}")

    model, device_obj = load_model_checkpoint(
        checkpoint_path=checkpoint_path,
        input_dim_1m=None,
        input_dim_5m=None,
        device=device,
    )

    records = predict_sequences(
        model=model,
        X1=X1,
        X5=X5,
        timestamps=timestamps,
        device=device_obj,
        batch_size=batch_size,
        center_predictions=center_predictions,
        clip_predictions=clip_predictions,
        clip_limit=clip_limit,
    )

    if use_execution_filters:
        records = apply_execution_filters(records)
    else:
        records = generate_trading_signals(records)

    csv_path = write_predictions_to_csv(records, output_csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV was not created: {csv_path}")

    return csv_path