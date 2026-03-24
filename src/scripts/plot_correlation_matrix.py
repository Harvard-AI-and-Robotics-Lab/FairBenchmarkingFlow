"""
Plot a Pearson correlation matrix across key evaluation metrics.

Metrics: FID, IS, CLIP Score, Pick Score, MinMax Harmonic Mean (MMHM)

Usage:
    python src/scripts/plot_correlation_matrix.py
    python src/scripts/plot_correlation_matrix.py --input data/imagenet_results.csv --output outputs/correlation_matrix.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_INPUT  = Path(__file__).resolve().parents[2] / "data" / "imagenet_results.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "correlation_matrix.png"

METRICS = ["FID", "IS", "CLIP Score", "Pick Score", "MinMax Harmonic Mean"]
LABELS  = ["FID", "IS", "CLIP Score", "Pick Score", "MMHM"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metric correlation matrix.")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    missing = [c for c in METRICS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    corr = df[METRICS].corr(method="pearson")
    corr.index   = LABELS
    corr.columns = LABELS

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # mask upper triangle

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
    )

    ax.set_title("Metric Correlation Matrix", fontsize=13, pad=12)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.output}")

    print("\nCorrelation matrix:")
    print(corr.to_string())


if __name__ == "__main__":
    main()
