import unittest

import numpy as np

from t790m_radiomics.reliability import icc_2_1


class TestReliability(unittest.TestCase):
    def test_perfect_agreement(self):
        values = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
        self.assertAlmostEqual(icc_2_1(values), 1.0)


if __name__ == "__main__":
    unittest.main()
