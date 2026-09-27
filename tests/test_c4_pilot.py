from __future__ import annotations

import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_c4_pilot import QUOTAS, SEED  # noqa: E402


class C4PilotTest(unittest.TestCase):
    def test_frozen_sampling_contract(self) -> None:
        self.assertEqual(SEED, 20260927)
        self.assertEqual(QUOTAS, {"A": (50, 50), "B": (47, 47)})
        self.assertEqual(sum(sum(value) for value in QUOTAS.values()) + 6, 200)


if __name__ == "__main__":
    unittest.main()
