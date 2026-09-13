"""Leakage-aware Mann-Whitney + L1-logistic radiomics modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MannWhitneySelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def fit(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        p_values = []
        for index in range(x.shape[1]):
            try:
                p = mannwhitneyu(x[y == 0, index], x[y == 1, index], alternative="two-sided").pvalue
            except ValueError:
                p = 1.0
            p_values.append(p)
        self.p_values_ = np.asarray(p_values)
        self.support_ = self.p_values_ < self.alpha
        if not np.any(self.support_):
            self.support_[int(np.nanargmin(self.p_values_))] = True
        return self

    def transform(self, x):
        return np.asarray(x)[:, self.support_]


def build_pipeline(inner_folds: int, seed: int, alpha: float) -> Pipeline:
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("univariate", MannWhitneySelector(alpha=alpha)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegressionCV(
                    Cs=np.logspace(-4, 4, 30),
                    cv=inner,
                    penalty="l1",
                    solver="liblinear",
                    scoring="roc_auc",
                    class_weight="balanced",
                    max_iter=10_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def youden_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def metrics(y_true: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--id-column", default="case_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--exclude", nargs="*", default=["batch", "split", "rater"])
    parser.add_argument("--icc-table", type=Path)
    parser.add_argument("--icc-threshold", type=float, default=0.80)
    parser.add_argument("--mann-whitney-alpha", type=float, default=0.05)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    y = frame[args.label_column].to_numpy(dtype=int)
    excluded = {args.id_column, args.label_column, *args.exclude}
    feature_names = [column for column in frame.select_dtypes(include=[np.number]).columns if column not in excluded]
    if args.icc_table:
        icc = pd.read_csv(args.icc_table)
        retained = set(icc.loc[icc["icc_2_1"] > args.icc_threshold, "feature"].astype(str))
        feature_names = [name for name in feature_names if name in retained]
    if not feature_names:
        raise ValueError("No eligible numeric features remain.")
    x = frame[feature_names].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    min_class = int(np.bincount(y).min())
    outer_folds, inner_folds = min(args.outer_folds, min_class), min(args.inner_folds, max(2, min_class - 1))
    if outer_folds < 2:
        raise ValueError("Both outcome classes need at least two samples.")

    probabilities = np.zeros(len(frame), dtype=float)
    predictions = np.zeros(len(frame), dtype=int)
    fold_ids = np.zeros(len(frame), dtype=int)
    selected_counts = {name: 0 for name in feature_names}
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y), start=1):
        model = build_pipeline(inner_folds, args.seed + fold, args.mann_whitney_alpha)
        model.fit(x[train_idx], y[train_idx])
        train_probability = model.predict_proba(x[train_idx])[:, 1]
        threshold = youden_threshold(y[train_idx], train_probability)
        probabilities[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        predictions[test_idx] = (probabilities[test_idx] >= threshold).astype(int)
        fold_ids[test_idx] = fold
        support = model.named_steps["univariate"].support_
        for name in np.asarray(feature_names)[support]:
            selected_counts[str(name)] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            args.id_column: frame[args.id_column],
            "label": y,
            "probability": probabilities,
            "prediction": predictions,
            "fold": fold_ids,
        }
    ).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics(y, probabilities, predictions), indent=2), encoding="utf-8")
    pd.DataFrame(
        {"feature": feature_names, "outer_fold_selection_count": [selected_counts[name] for name in feature_names]}
    ).sort_values("outer_fold_selection_count", ascending=False).to_csv(
        args.output_dir / "feature_stability.csv", index=False
    )

    final_model = build_pipeline(inner_folds, args.seed, args.mann_whitney_alpha).fit(x, y)
    joblib.dump({"model": final_model, "feature_names": feature_names}, args.output_dir / "final_model.joblib")
    if args.external_csv:
        external = pd.read_csv(args.external_csv)
        external_x = external[feature_names].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        external_probability = final_model.predict_proba(external_x)[:, 1]
        result = pd.DataFrame({args.id_column: external[args.id_column], "probability": external_probability})
        if args.label_column in external:
            result["label"] = external[args.label_column].to_numpy(dtype=int)
        result.to_csv(args.output_dir / "external_predictions.csv", index=False)


if __name__ == "__main__":
    main()
