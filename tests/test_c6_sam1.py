from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # The lightweight manifest-only environment intentionally has no NumPy.
    np = None

if np is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_sam1_pilot import iou_numpy


@unittest.skipIf(np is None, "NumPy is exercised in the DGX SAM environment")
class C6Sam1Test(unittest.TestCase):
    def test_numpy_iou(self) -> None:
        prediction = np.array([[1, 1], [0, 0]], dtype=bool)
        reference = np.array([[0, 1], [1, 0]], dtype=bool)
        self.assertAlmostEqual(iou_numpy(prediction, reference), 1 / 3)

    def test_empty_prediction(self) -> None:
        self.assertEqual(iou_numpy(np.zeros((2, 2), bool), np.eye(2, dtype=bool)), 0.0)


if __name__ == "__main__":
    unittest.main()
