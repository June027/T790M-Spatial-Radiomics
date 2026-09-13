"""ROI-restricted local entropy for 3D MRI volumes."""

from __future__ import annotations

import numpy as np


def quantize_uint8(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linearly quantize ROI intensities to 0-255 for local histograms."""
    roi = image[mask]
    if roi.size == 0:
        raise ValueError("The ROI mask is empty.")
    low, high = float(np.min(roi)), float(np.max(roi))
    result = np.zeros(image.shape, dtype=np.uint8)
    if high > low:
        result[mask] = np.clip(np.rint((roi - low) * 255.0 / (high - low)), 0, 255).astype(np.uint8)
    return result


def local_entropy_2d(image: np.ndarray, mask: np.ndarray, neighborhood: int = 9) -> np.ndarray:
    """Compute natural-log entropy in an odd square neighborhood for ROI pixels."""
    if image.shape != mask.shape or image.ndim != 2:
        raise ValueError("image and mask must be same-shaped 2D arrays.")
    if neighborhood < 1 or neighborhood % 2 == 0:
        raise ValueError("neighborhood must be a positive odd integer.")
    image_u8 = quantize_uint8(image, mask.astype(bool))
    radius = neighborhood // 2
    entropy_map = np.zeros(image.shape, dtype=np.float32)
    for row, column in np.argwhere(mask):
        r0, r1 = max(0, row - radius), min(image.shape[0], row + radius + 1)
        c0, c1 = max(0, column - radius), min(image.shape[1], column + radius + 1)
        window_mask = mask[r0:r1, c0:c1].astype(bool)
        values = image_u8[r0:r1, c0:c1][window_mask]
        counts = np.bincount(values, minlength=256)
        probabilities = counts[counts > 0] / max(len(values), 1)
        entropy_map[row, column] = float(-np.sum(probabilities * np.log(probabilities)))
    return entropy_map


def local_entropy_volume(image: np.ndarray, mask: np.ndarray, neighborhood: int = 9) -> np.ndarray:
    """Apply local entropy slice by slice to a ZxYxX volume."""
    if image.shape != mask.shape or image.ndim != 3:
        raise ValueError("image and mask must be same-shaped 3D arrays.")
    output = np.zeros(image.shape, dtype=np.float32)
    for index in range(image.shape[0]):
        if np.any(mask[index]):
            output[index] = local_entropy_2d(image[index], mask[index], neighborhood)
    return output


def zscore_roi(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.float32)
    roi = values[mask]
    scale = float(np.std(roi))
    result[mask] = (roi - float(np.mean(roi))) / (scale if scale > 0 else 1.0)
    return result
