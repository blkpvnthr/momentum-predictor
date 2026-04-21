#!/usr/bin/env python3

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


SYMBOLS = [
    "QQQ",
    "TQQQ",
    "QBTS",
    "AAPL",
    "QTUM",
    "RGTI",
    "OPEN",
    "TSLL",
    "SOXL",
    "TECL",
    "APLD",
    "ASTS",
    "IONQ",
    "QUBT",
]

TARGET = "QQQ"
START_DATE = "2024-01-01"


# -------------------------
# DATA DOWNLOAD
# -------------------------
def download_prices(symbols):
    df = yf.download(
        symbols,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    closes = []
    for s in symbols:
        if s in df.columns.get_level_values(0):
            closes.append(df[s]["Close"].rename(s))

    return pd.concat(closes, axis=1).dropna(how="all")


# -------------------------
# CORRELATION + METRICS
# -------------------------
def compute_metrics(df):
    returns = df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    target = returns[TARGET]

    rows = []

    for col in returns.columns:
        if col == TARGET:
            continue

        aligned = pd.concat([target, returns[col]], axis=1).dropna()
        aligned.columns = [TARGET, col]

        corr = aligned[TARGET].corr(aligned[col])

        same_direction = (
            np.sign(aligned[TARGET]) == np.sign(aligned[col])
        ).mean()

        # combined ranking score (you can tweak weights)
        score = (0.7 * corr) + (0.3 * same_direction)

        rows.append({
            "symbol": col,
            "correlation": corr,
            "same_direction_rate": same_direction,
            "score": score,
        })

    result = pd.DataFrame(rows).sort_values(by="score", ascending=False)

    return result, returns.corr()


# -------------------------
# HEATMAP
# -------------------------
def plot_heatmap(corr):
    fig, ax = plt.subplots(figsize=(12, 10))

    cax = ax.imshow(corr.values)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)

    ax.set_title("Correlation Heatmap (Daily Returns)")
    fig.colorbar(cax)

    plt.tight_layout()
    plt.show()


# -------------------------
# MAIN
# -------------------------
def main():
    print("[download] fetching data...")
    df = download_prices(SYMBOLS)

    print("[compute] calculating correlations + rankings...")
    ranked, corr = compute_metrics(df)

    # 🔥 TOP 10 CLOSEST MOVERS
    print("\n===== TOP 10 CLOSEST MOVERS TO QQQ =====\n")
    print(ranked.head(10).to_string(index=False, float_format="%.4f"))

    # 🔥 HEATMAP
    print("\n[plot] rendering heatmap...")
    plot_heatmap(corr)


if __name__ == "__main__":
    main()