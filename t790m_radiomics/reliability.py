"""Estimate ICC(2,1) from repeated radiomics measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def icc_2_1(values: np.ndarray) -> float:
    """Two-way random-effects, absolute-agreement, single-measure ICC."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return float("nan")
    n, k = values.shape
    grand = values.mean()
    row_means, column_means = values.mean(axis=1), values.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((column_means - grand) ** 2) / (k - 1)
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator != 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--rater-column", default="rater")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    excluded = {args.case_column, args.rater_column, "label", "batch"}
    features = [column for column in frame.select_dtypes(include=[np.number]).columns if column not in excluded]
    rows = []
    for feature in features:
        pivot = frame.pivot(index=args.case_column, columns=args.rater_column, values=feature).dropna()
        value = icc_2_1(pivot.to_numpy())
        rows.append({"feature": feature, "icc_2_1": value, "retained": bool(value > args.threshold)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("icc_2_1", ascending=False).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
