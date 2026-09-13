"""Export linear SHAP values for a fitted radiomics signature."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    pipeline, feature_names = bundle["model"], bundle["feature_names"]
    frame = pd.read_csv(args.input)
    x = frame[feature_names].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    imputed = pipeline.named_steps["imputer"].transform(x)
    support = pipeline.named_steps["univariate"].support_
    selected_names = np.asarray(feature_names)[support]
    scaled = pipeline.named_steps["scaler"].transform(imputed[:, support])
    classifier = pipeline.named_steps["classifier"]
    explanation = shap.LinearExplainer(classifier, scaled)(scaled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "shap_values.npy", explanation.values)
    pd.DataFrame(
        {"feature": selected_names, "mean_absolute_shap": np.abs(explanation.values).mean(axis=0)}
    ).sort_values("mean_absolute_shap", ascending=False).to_csv(args.output_dir / "shap_importance.csv", index=False)


if __name__ == "__main__":
    main()
