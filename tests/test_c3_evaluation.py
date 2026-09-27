from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_masks import binary_mask_iou, summarize_ious  # noqa: E402


class C3EvaluationTest(unittest.TestCase):
    def test_perfect_iou(self) -> None:
        mask = [[0, 1], [1, 0]]
        self.assertEqual(binary_mask_iou(mask, mask), 1.0)

    def test_partial_iou(self) -> None:
        prediction = [[1, 1], [0, 0]]
        reference = [[0, 1], [1, 0]]
        self.assertAlmostEqual(binary_mask_iou(prediction, reference), 1 / 3)

    def test_empty_prediction_is_zero(self) -> None:
        self.assertEqual(binary_mask_iou([[0, 0]], [[1, 0]]), 0.0)

    def test_empty_reference_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            binary_mask_iou([[0]], [[0]])

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            binary_mask_iou([[1, 0]], [[1]])

    def test_thresholds_are_inclusive(self) -> None:
        summary = summarize_ious([0.0, 0.5, 0.7, 0.8, 0.9, 1.0])
        self.assertEqual(summary["count"], 6)
        self.assertEqual(summary["failure_count_below_0.20"], 1)
        self.assertEqual(summary["failure_rate_below_0.20"], 1 / 6)
        self.assertEqual(summary["success_rates"]["iou_at_least_0.50"], 5 / 6)
        self.assertEqual(summary["success_rates"]["iou_at_least_0.70"], 4 / 6)
        self.assertEqual(summary["success_rates"]["iou_at_least_0.80"], 3 / 6)
        self.assertEqual(summary["success_rates"]["iou_at_least_0.90"], 2 / 6)

    def test_failure_threshold_is_strictly_below(self) -> None:
        summary = summarize_ious([0.0, 0.1999, 0.2])
        self.assertEqual(summary["failure_count_below_0.20"], 2)


if __name__ == "__main__":
    unittest.main()
