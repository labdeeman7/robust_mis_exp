from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_c5_prompts import make_prompt, shifted_entry, visible_points  # noqa: E402


class C5PromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pose = {"nodes": [[-20, 40], [10, 40], [50, 30], None],
                     "tags": ["occluded", "occluded", "visible", "missing"],
                     "edges": [[0, 1], [1, 2], [1, 3]]}

    def test_only_visible_and_in_bounds_points_survive(self) -> None:
        self.assertEqual(visible_points(self.pose, 100, 80), [[50.0, 30.0]])

    def test_occluded_points_never_enter_primary_prompt(self) -> None:
        prompt = make_prompt("p1_visible_points", self.pose, [], 100, 80)
        self.assertEqual(prompt["positive_points"], [[50.0, 30.0]])

    def test_entry_shift_is_explicit_ablation(self) -> None:
        shifted = shifted_entry(self.pose, 100, 80, inward=10)
        self.assertIsNotNone(shifted)
        self.assertGreaterEqual(shifted[0], 0)
        ordinary = make_prompt("p1_visible_points", self.pose, [], 100, 80)
        ablation = make_prompt("p7_shifted_entry_ablation", self.pose, [], 100, 80)
        self.assertIsNone(ordinary["shifted_entry_point"])
        self.assertIsNotNone(ablation["shifted_entry_point"])

    def test_negative_points_also_require_visibility(self) -> None:
        other = {"nodes": [[20, 20], [30, 30]], "tags": ["occluded", "visible"], "edges": []}
        prompt = make_prompt("p5_visible_points_negatives", self.pose, [other], 100, 80)
        self.assertEqual(prompt["negative_points"], [[30.0, 30.0]])


if __name__ == "__main__":
    unittest.main()
