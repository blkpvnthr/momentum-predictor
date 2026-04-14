from __future__ import annotations

import json
from pathlib import Path

import torch

from momentum_predictor.pipeline import run_pipeline
from momentum_predictor.model import (
    get_device,
    train_tabular_baseline,
    train_tcn_baseline,
    train_model,
)
from momentum_predictor.inference import run_inference_to_csv


# =========================================================
# PATHS
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
SIGNAL_DIR = OUTPUT_DIR / "signals"

MODEL_PATH = MODEL_DIR / "best_model2.pt"
OUTPUT_CSV = SIGNAL_DIR / "predictions.csv"
TRAIN_SUMMARY_PATH = OUTPUT_DIR / "train_summary.json"


# =========================================================
# CONFIG
# =========================================================
PIPELINE_SYMBOL = "QQQ"
PIPELINE_START = "2026-03-01"
PIPELINE_END = "2026-04-01"
USE_REGIME_CACHE = True

INFERENCE_DEVICE = "auto"
INFERENCE_BATCH_SIZE = 512
CENTER_PREDICTIONS = True
USE_EXECUTION_FILTERS = True


# =========================================================
# HELPERS
# =========================================================
def get_cpu_device() -> torch.device:
    return torch.device("cpu")

def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    model_path: Path,
    *,
    x1_shape: tuple[int, ...],
    x5_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    pipeline_symbol: str,
    pipeline_start: str,
    pipeline_end: str,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "x1_shape": list(x1_shape),
            "x5_shape": list(x5_shape),
            "y_shape": list(y_shape),
            "input_dim_1m": int(x1_shape[-1]),
            "input_dim_5m": int(x5_shape[-1]),
            "seq_len_1m": int(x1_shape[1]),
            "seq_len_5m": int(x5_shape[1]),
            "pipeline_symbol": pipeline_symbol,
            "pipeline_start": pipeline_start,
            "pipeline_end": pipeline_end,
        },
    }

    torch.save(checkpoint, model_path)


def write_train_summary(
    path: Path,
    *,
    x1_shape: tuple[int, ...],
    x5_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    timestamps_count: int,
    model_path: Path,
    output_csv: Path,
) -> None:
    summary = {
        "x1_shape": list(x1_shape),
        "x5_shape": list(x5_shape),
        "y_shape": list(y_shape),
        "timestamps_count": int(timestamps_count),
        "model_path": str(model_path),
        "predictions_csv": str(output_csv),
    }
    path.write_text(json.dumps(summary, indent=2))



# =========================================================
# MAIN
# =========================================================
def main() -> None:
    ensure_dirs()

    print("=== RUNNING PIPELINE ===")
    X1, X5, y, timestamps = run_pipeline(
        symbol=PIPELINE_SYMBOL,
        start=PIPELINE_START,
        end=PIPELINE_END,
        use_regime_cache=USE_REGIME_CACHE,
    )

    print("\n=== PIPELINE SUMMARY ===")
    print(f"X1 shape: {X1.shape}")
    print(f"X5 shape: {X5.shape}")
    print(f"y shape:  {y.shape}")
    print(f"timestamps: {len(timestamps)}")

    if len(X1) == 0 or len(X5) == 0 or len(y) == 0:
        raise RuntimeError("Pipeline returned empty arrays; cannot continue training.")

    print("\n=== TRAINING BASELINES (CPU) ===")

    try:
        print("[train] tabular baseline (CPU)...")
        train_tabular_baseline(X1, y, device="cpu")
    except Exception as exc:
        print(f"[train] tabular baseline failed: {exc}")

    try:
        print("[train] tcn baseline (CPU)...")
        train_tcn_baseline(X1, y, device="cpu")
    except Exception as exc:
        print(f"[train] tcn baseline failed: {exc}")

    print("\n=== TRAINING MAIN MODEL (GPU) ===")
    gpu_device = get_device("auto")
    print(f"[train] using device: {gpu_device}")

    model = train_model(X1, X5, y, device="auto")

    if model is None:
        raise RuntimeError("Main model training returned None; checkpoint not saved.")

    save_checkpoint(
        model=model,
        model_path=MODEL_PATH,
        x1_shape=X1.shape,
        x5_shape=X5.shape,
        y_shape=y.shape,
        pipeline_symbol=PIPELINE_SYMBOL,
        pipeline_start=PIPELINE_START,
        pipeline_end=PIPELINE_END,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint was not created: {MODEL_PATH}")

    print(f"[train] model saved -> {MODEL_PATH}")

    print("\n=== RUNNING INFERENCE ===")
    csv_path = run_inference_to_csv(
        checkpoint_path=MODEL_PATH,
        X1=X1,
        X5=X5,
        timestamps=timestamps,
        output_csv_path=OUTPUT_CSV,
        device=INFERENCE_DEVICE,
        batch_size=INFERENCE_BATCH_SIZE,
        center_predictions=CENTER_PREDICTIONS,
        use_execution_filters=USE_EXECUTION_FILTERS,
    )

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Predictions CSV was not created: {csv_path}")

    print(f"[inference] predictions saved -> {csv_path}")

    write_train_summary(
        path=TRAIN_SUMMARY_PATH,
        x1_shape=X1.shape,
        x5_shape=X5.shape,
        y_shape=y.shape,
        timestamps_count=len(timestamps),
        model_path=MODEL_PATH,
        output_csv=csv_path,
    )
    print(f"[train] summary saved -> {TRAIN_SUMMARY_PATH}")

    print("\n=== DONE ===")
    print(f"checkpoint: {MODEL_PATH}")
    print(f"predictions: {csv_path}")
    print(f"summary: {TRAIN_SUMMARY_PATH}")


if __name__ == "__main__":
    main()