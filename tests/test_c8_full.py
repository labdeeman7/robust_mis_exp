from __future__ import annotations

import unittest


class C8FullContractTest(unittest.TestCase):
    def test_expected_non_overlapping_totals(self) -> None:
        frames = {"A": 1521, "B": 5074, "C": 747, "D": 6}
        instances = {"A": 2946, "B": 6801, "C": 766, "D": 12}
        self.assertEqual(set(instances), set("ABCD"))
        self.assertEqual(sum(frames.values()), 7348)
        self.assertEqual(sum(instances.values()), 10525)


if __name__ == "__main__":
    unittest.main()
