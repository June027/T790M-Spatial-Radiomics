import unittest

import numpy as np

from t790m_radiomics.entropy import local_entropy_2d, zscore_roi
from t790m_radiomics.partition import patient_superpixels, select_population_k


class TestEntropyPartition(unittest.TestCase):
    def test_constant_roi_has_zero_entropy(self):
        image = np.ones((7, 7), dtype=float)
        mask = np.ones((7, 7), dtype=bool)
        result = local_entropy_2d(image, mask, neighborhood=3)
        self.assertTrue(np.allclose(result, 0.0))

    def test_roi_zscore(self):
        values = np.arange(16, dtype=float).reshape(4, 4)
        mask = np.zeros_like(values, dtype=bool)
        mask[1:3, 1:3] = True
        scaled = zscore_roi(values, mask)
        self.assertAlmostEqual(float(scaled[mask].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(scaled[mask].std()), 1.0, places=6)

    def test_partition_and_k_selection(self):
        image = np.arange(64, dtype=float).reshape(1, 8, 8)
        mask = np.ones_like(image, dtype=bool)
        entropy = np.flip(image, axis=2)
        labels, summaries = patient_superpixels(image, mask, entropy, n_superpixels=6, seed=42)
        self.assertEqual(labels.shape, image.shape)
        self.assertEqual(summaries.shape, (6, 2))
        best_k, scores = select_population_k(np.vstack([summaries, summaries + 3]), [2, 3], 42)
        self.assertIn(best_k, (2, 3))
        self.assertEqual(set(scores["k"]), {2, 3})


if __name__ == "__main__":
    unittest.main()
