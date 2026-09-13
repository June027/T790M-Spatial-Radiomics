"""Construct patient-level superpixels and population-level MRI subregions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from tqdm import tqdm

from .entropy import local_entropy_volume, zscore_roi


@dataclass
class CasePartition:
    case_id: str
    reference: sitk.Image
    roi_mask: np.ndarray
    superpixels: np.ndarray
    summaries: np.ndarray


def patient_superpixels(
    image: np.ndarray, mask: np.ndarray, entropy: np.ndarray, n_superpixels: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    roi = mask.astype(bool)
    features = np.column_stack([zscore_roi(image, roi)[roi], zscore_roi(entropy, roi)[roi]])
    n_clusters = min(n_superpixels, len(features))
    if n_clusters < 2:
        raise ValueError("At least two ROI voxels are required for partitioning.")
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(features)
    label_map = np.zeros(mask.shape, dtype=np.int16)
    label_map[roi] = labels + 1
    summaries = np.asarray([features[labels == index].mean(axis=0) for index in range(n_clusters)])
    return label_map, summaries


def select_population_k(features: np.ndarray, candidates: list[int], seed: int):
    rows = []
    for k in candidates:
        if k < 2 or k >= len(features):
            continue
        labels = KMeans(n_clusters=k, random_state=seed, n_init=50).fit_predict(features)
        rows.append(
            {
                "k": k,
                "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
                "silhouette": float(silhouette_score(features, labels)),
            }
        )
    if not rows:
        raise ValueError("No valid population-level K candidate.")
    frame = pd.DataFrame(rows)
    for column in ("calinski_harabasz", "silhouette"):
        values = frame[column].to_numpy()
        frame[f"{column}_scaled"] = (values - values.min()) / max(float(np.ptp(values)), 1e-12)
    frame["joint_score"] = frame["calinski_harabasz_scaled"] + frame["silhouette_scaled"]
    best_k = int(frame.sort_values(["joint_score", "k"], ascending=[False, True]).iloc[0]["k"])
    return best_k, frame


def population_labels(features: np.ndarray, k: int, method: str, seed: int) -> np.ndarray:
    if method == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(features)
    return KMeans(n_clusters=k, random_state=seed, n_init=100).fit_predict(features)


def process_sequence(rows: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    cases: list[CasePartition] = []
    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=f"Superpixels {rows.iloc[0]['sequence']}"):
        image_obj, mask_obj = sitk.ReadImage(str(row.image)), sitk.ReadImage(str(row.mask))
        image, mask = sitk.GetArrayFromImage(image_obj).astype(np.float32), sitk.GetArrayFromImage(mask_obj) > 0
        if image.shape != mask.shape:
            raise ValueError(f"Shape mismatch for {row.case_id}: image {image.shape}, mask {mask.shape}")
        entropy = local_entropy_volume(image, mask, args.neighborhood)
        superpixels, summaries = patient_superpixels(image, mask, entropy, args.superpixels, args.seed)
        cases.append(CasePartition(str(row.case_id), image_obj, mask, superpixels, summaries))

    pooled = np.concatenate([case.summaries for case in cases], axis=0)
    best_k, scores = select_population_k(pooled, args.candidate_k, args.seed)
    pooled_labels = population_labels(pooled, best_k, args.population_method, args.seed)
    sequence = str(rows.iloc[0]["sequence"])
    sequence_dir = output_dir / sequence
    sequence_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(sequence_dir / "cluster_selection.csv", index=False)
    (sequence_dir / "partition_summary.json").write_text(
        json.dumps({"sequence": sequence, "selected_k": best_k, "population_method": args.population_method}, indent=2),
        encoding="utf-8",
    )

    offset = 0
    for case in cases:
        labels = pooled_labels[offset : offset + len(case.summaries)] + 1
        offset += len(case.summaries)
        subregions = np.zeros(case.superpixels.shape, dtype=np.uint8)
        for local_label, population_label in enumerate(labels, start=1):
            subregions[case.superpixels == local_label] = population_label
        output = sitk.GetImageFromArray(subregions)
        output.CopyInformation(case.reference)
        sitk.WriteImage(output, str(sequence_dir / f"{case.case_id}_subregions.nrrd"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighborhood", type=int, default=9)
    parser.add_argument("--superpixels", type=int, default=30)
    parser.add_argument("--candidate-k", type=int, nargs="+", default=list(range(2, 11)))
    parser.add_argument("--population-method", choices=("kmeans", "ward"), default="kmeans")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    required = {"case_id", "sequence", "image", "mask"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain columns {sorted(required)}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for _, rows in manifest.groupby("sequence", sort=True):
        process_sequence(rows.reset_index(drop=True), args.output_dir, args)


if __name__ == "__main__":
    main()
