"""
Compute composite scores for generative model benchmarks.

Reads a CSV with FID, IS, CLIP Score, and Pick Score columns and appends
the MinMax Harmonic Mean composite score computed across all rows.

Usage:
    python src/scripts/compute_composite_score.py
    python src/scripts/compute_composite_score.py --input data/imagenet_results.csv --output data/imagenet_results_scored.csv

    # Save the min/max bounds used for normalization:
    python src/scripts/compute_composite_score.py --save-bounds data/imagenet_bounds.json

    # Reuse previously saved bounds (for generalization to new evaluation sets):
    python src/scripts/compute_composite_score.py --input data/new_results.csv --reuse-bounds data/imagenet_bounds.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import sys

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.composite_metric import minmax_harmonic_mean

DEFAULT_INPUT  = Path(__file__).resolve().parents[2] / "data" / "imagenet_results.csv"
DEFAULT_OUTPUT = DEFAULT_INPUT  # overwrite in-place by default

# Known aliases for each canonical column name.
# Keys are canonical; values are lists of alternative names (case-insensitive).
_COLUMN_ALIASES: dict[str, list[str]] = {
    "FID":        ["fid"],
    "IS":         ["is_mean", "inception_score", "is"],
    "CLIP Score": ["clip_score", "clip"],
    "Pick Score": ["pick_score", "pickscore"],
    "Model":      ["model_name", "run_name", "name"],
    "Steps":      ["num_steps", "step", "nsteps"],
}


def _remap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using known aliases.

    Performs case-insensitive matching so 'FID', 'fid', 'Fid' all resolve.
    Prints a warning for every column that is remapped.
    """
    lower_to_actual = {c.lower(): c for c in df.columns}
    rename_map: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue  # already correct
        for alias in aliases:
            if alias.lower() in lower_to_actual:
                actual = lower_to_actual[alias.lower()]
                rename_map[actual] = canonical
                print(f"  [column remap] '{actual}' -> '{canonical}'")
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Append composite scores to benchmark CSV.")
    parser.add_argument("--input",        type=Path, default=DEFAULT_INPUT,  help="Path to input CSV")
    parser.add_argument("--output",       type=Path, default=DEFAULT_OUTPUT, help="Path to output CSV (default: overwrite input)")
    parser.add_argument("--eps",          type=float, default=1e-3,          help="Epsilon for harmonic mean (default: 1e-3)")
    parser.add_argument("--save-bounds",  type=Path, default=None,           help="Save the min/max normalization bounds to this JSON file")
    parser.add_argument("--reuse-bounds", type=Path, default=None,           help="Load previously saved bounds JSON instead of recomputing from data")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = _remap_columns(df)

    required = {"FID", "IS", "CLIP Score", "Pick Score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    fid  = df["FID"].tolist()
    is_  = df["IS"].tolist()
    clip = df["CLIP Score"].tolist()
    pick = df["Pick Score"].tolist()

    # Load pre-computed bounds if requested
    bounds = None
    if args.reuse_bounds is not None:
        with open(args.reuse_bounds) as f:
            bounds = json.load(f)
        print(f"Reusing normalization bounds from {args.reuse_bounds}")

    SCORE_COLS = ["Rank Geometric Mean", "Rank Harmonic Mean", "MinMax Harmonic Mean"]

    # Drop existing score columns so they are cleanly overwritten
    df.drop(columns=[c for c in SCORE_COLS if c in df.columns], inplace=True)

    def sig_fig(arr, n=5):
        import numpy as np
        return [float(f"{v:.{n}g}") for v in arr]

    scores, bounds_used = minmax_harmonic_mean(fid, is_, clip, pick, eps=args.eps, bounds=bounds)
    df["MinMax Harmonic Mean"] = sig_fig(scores)

    # Persist bounds so they can be reused for future evaluation sets
    if args.save_bounds is not None:
        args.save_bounds.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_bounds, "w") as f:
            json.dump(bounds_used, f, indent=2)
        print(f"Saved normalization bounds -> {args.save_bounds}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows with composite scores -> {args.output}")

    score_cols = [c for c in ["Model", "Steps", "MinMax Harmonic Mean"] if c in df.columns]
    print(df[score_cols].to_string(index=False))


if __name__ == "__main__":
    main()
