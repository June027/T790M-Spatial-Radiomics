"""ComBat harmonization wrapper for multicenter radiomics tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationApply, harmonizationLearn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-column", default="batch")
    parser.add_argument("--preserve", nargs="*", default=["case_id", "label", "batch"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.batch_column not in frame:
        raise ValueError(f"Missing batch column: {args.batch_column}")
    feature_columns = [
        column for column in frame.select_dtypes(include=[np.number]).columns if column not in set(args.preserve)
    ]
    if not feature_columns:
        raise ValueError("No numeric radiomics features found.")
    matrix = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median()).to_numpy(dtype=float)
    covariates = pd.DataFrame({"SITE": frame[args.batch_column].astype(str)})
    model, _ = harmonizationLearn(matrix, covariates)
    adjusted = harmonizationApply(matrix, covariates, model)
    output = frame.copy()
    output.loc[:, feature_columns] = adjusted
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
