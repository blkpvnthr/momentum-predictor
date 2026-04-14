from pathlib import Path

from momentum_predictor.pipeline import run_pipeline
from momentum_predictor.inference import run_inference_to_csv

PROJECT_ROOT = Path("/Users/blkpvnthr/Desktop/momentum-predictor")
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_model.pt"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "signals" / "predictions.csv"

print("model exists:", MODEL_PATH.exists())
print("output exists before:", OUTPUT_CSV.exists())

X1, X5, y, timestamps = run_pipeline()

csv_path = run_inference_to_csv(
    checkpoint_path=MODEL_PATH,
    X1=X1,
    X5=X5,
    timestamps=timestamps,
    output_csv_path=OUTPUT_CSV,
    device="auto",
    batch_size=512,
    center_predictions=True,
    use_execution_filters=True,
)

print("saved to:", csv_path)
print("output exists after:", OUTPUT_CSV.exists())