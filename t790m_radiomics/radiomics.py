"""Extract PyRadiomics features from spatial subregions and whole tumors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm


def clean_features(result: dict[str, object]) -> dict[str, float]:
    cleaned = {}
    for name, value in result.items():
        if name.startswith("diagnostics_"):
            continue
        try:
            cleaned[name] = float(np.asarray(value).squeeze())
        except (TypeError, ValueError):
            continue
    return cleaned


def extract_regions(image: sitk.Image, mask: sitk.Image, extractor) -> dict[str, float]:
    mask_array = sitk.GetArrayFromImage(mask)
    labels = sorted(int(value) for value in np.unique(mask_array) if value > 0)
    output: dict[str, float] = {}
    for label in labels:
        for feature, value in clean_features(extractor.execute(image, mask, label=label)).items():
            output[f"S{label}_{feature}"] = value
    whole_array = (mask_array > 0).astype(np.uint8)
    whole_mask = sitk.GetImageFromArray(whole_array)
    whole_mask.CopyInformation(mask)
    for feature, value in clean_features(extractor.execute(image, whole_mask, label=1)).items():
        output[f"Whole_{feature}"] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/radiomics.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    required = {"case_id", "sequence", "image", "subregion_mask"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain columns {sorted(required)}.")
    extractor = featureextractor.RadiomicsFeatureExtractor(str(args.config))
    records: dict[str, dict[str, object]] = {}
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Radiomics"):
        case_id, sequence = str(row.case_id), str(row.sequence)
        record = records.setdefault(case_id, {"case_id": case_id})
        for column in ("batch", "label"):
            if hasattr(row, column):
                record[column] = getattr(row, column)
        image, mask = sitk.ReadImage(str(row.image)), sitk.ReadImage(str(row.subregion_mask))
        for name, value in extract_regions(image, mask, extractor).items():
            record[f"{sequence}_{name}"] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records.values()).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
