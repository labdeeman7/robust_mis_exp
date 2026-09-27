from __future__ import annotations

import unittest


class C8FullContractTest(unittest.TestCase):
    def test_expected_non_overlapping_totals(self) -> None:
        frames = {"A": 6595, "B": 747, "C": 6}
        instances = {"A": 9178, "B": 1335, "C": 12}
        self.assertEqual(sum(frames.values()), 7348)
        self.assertEqual(sum(instances.values()), 10525)


if __name__ == "__main__":
    unittest.main()
