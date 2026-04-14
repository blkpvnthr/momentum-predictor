import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# PATHS
# =========================================================
INPUT_PATH = Path("outputs/signals/predictions.csv")
OUTPUT_DIR = Path("outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# =========================================================
# BASIC INFO
# =========================================================
print("\n=== BASIC STATS ===")
basic_stats = df.describe()
print(basic_stats)

print("\n=== SIGNAL COUNTS ===")
signal_counts = df["signal"].value_counts()
print(signal_counts)

print("\n=== CONFIDENCE STATS ===")
confidence_stats = df["confidence"].describe()
print(confidence_stats)

# =========================================================
# TOP CONFIDENCE
# =========================================================
top_conf = df.sort_values("confidence", ascending=False).head(50)
print("\n=== TOP 50 CONFIDENCE ===")
print(top_conf)

# =========================================================
# RETURN DISTRIBUTION
# =========================================================
ret_stats = {
    "5m_mean": df["pred_return_5m"].mean(),
    "5m_std": df["pred_return_5m"].std(),
    "15m_mean": df["pred_return_15m"].mean(),
    "15m_std": df["pred_return_15m"].std(),
    "30m_mean": df["pred_return_30m"].mean(),
    "30m_std": df["pred_return_30m"].std(),
}

print("\n=== RETURN DISTRIBUTION ===")
print(ret_stats)

# =========================================================
# LONG / SHORT ANALYSIS
# =========================================================
longs = df[df["signal"] == "LONG"]
shorts = df[df["signal"] == "SHORT"]

long_summary = {}
short_summary = {}

print("\n=== LONG SIGNALS ===")
long_summary["count"] = len(longs)
print("count:", long_summary["count"])

if len(longs) > 0:
    long_summary["mean_return_15m"] = longs["pred_return_15m"].mean()
    long_summary["avg_confidence"] = longs["confidence"].mean()

    print("mean return:", long_summary["mean_return_15m"])
    print("avg confidence:", long_summary["avg_confidence"])

    top_longs = longs.sort_values("pred_return_15m", ascending=False).head(50)
    print("\nTop LONG trades:")
    print(top_longs)
else:
    top_longs = pd.DataFrame()

print("\n=== SHORT SIGNALS ===")
short_summary["count"] = len(shorts)
print("count:", short_summary["count"])

if len(shorts) > 0:
    short_summary["mean_return_15m"] = shorts["pred_return_15m"].mean()
    short_summary["avg_confidence"] = shorts["confidence"].mean()

    print("mean return:", short_summary["mean_return_15m"])
    print("avg confidence:", short_summary["avg_confidence"])

    top_shorts = shorts.sort_values("pred_return_15m").head(50)
    print("\nTop SHORT trades:")
    print(top_shorts)
else:
    top_shorts = pd.DataFrame()

# =========================================================
# DECILES
# =========================================================
top_threshold = df["pred_return_15m"].quantile(0.9)
bottom_threshold = df["pred_return_15m"].quantile(0.1)

top = df[df["pred_return_15m"] >= top_threshold]
bottom = df[df["pred_return_15m"] <= bottom_threshold]

decile_summary = {
    "top_count": len(top),
    "top_mean_return": top["pred_return_15m"].mean(),
    "top_avg_confidence": top["confidence"].mean(),
    "bottom_count": len(bottom),
    "bottom_mean_return": bottom["pred_return_15m"].mean(),
    "bottom_avg_confidence": bottom["confidence"].mean(),
}

print("\n=== DECILES ===")
print(decile_summary)

# =========================================================
# ALIGNMENT
# =========================================================
aligned_longs = longs[longs["pred_return_15m"] > 0]
aligned_shorts = shorts[shorts["pred_return_15m"] < 0]

alignment = {}

print("\n=== SIGNAL ALIGNMENT ===")

if len(longs) > 0:
    alignment["long_alignment"] = len(aligned_longs) / len(longs)
    print(f"LONG alignment: {alignment['long_alignment']:.2%}")

if len(shorts) > 0:
    alignment["short_alignment"] = len(aligned_shorts) / len(shorts)
    print(f"SHORT alignment: {alignment['short_alignment']:.2%}")

# =========================================================
# SAVE OUTPUTS
# =========================================================

# 1. Summary CSV
summary = {
    **ret_stats,
    **decile_summary,
    **alignment,
    "long_count": long_summary.get("count", 0),
    "long_mean_return": long_summary.get("mean_return_15m", np.nan),
    "long_avg_conf": long_summary.get("avg_confidence", np.nan),
    "short_count": short_summary.get("count", 0),
    "short_mean_return": short_summary.get("mean_return_15m", np.nan),
    "short_avg_conf": short_summary.get("avg_confidence", np.nan),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

# 2. Core tables
basic_stats.to_csv(OUTPUT_DIR / "basic_stats.csv")
confidence_stats.to_csv(OUTPUT_DIR / "confidence_stats.csv")

# 3. Trade breakdowns
top_conf.to_csv(OUTPUT_DIR / "top_confidence.csv", index=False)
top_longs.to_csv(OUTPUT_DIR / "top_longs.csv", index=False)
top_shorts.to_csv(OUTPUT_DIR / "top_shorts.csv", index=False)

# 4. Deciles
top.to_csv(OUTPUT_DIR / "top_decile.csv", index=False)
bottom.to_csv(OUTPUT_DIR / "bottom_decile.csv", index=False)

print(f"\n✅ Analysis saved to: {OUTPUT_DIR.resolve()}")