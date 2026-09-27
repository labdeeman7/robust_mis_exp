from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
import zlib
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_robust_mips_archive import audit  # noqa: E402


def png(width: int, height: int, value: int = 0) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    pixels = b"".join(b"\x00" + bytes([value]) * width for _ in range(height))
    return signature + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


class ArchiveAuditTest(unittest.TestCase):
    def test_complete_frame_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.zip"
            frame = "wrapper/training/rectal/procedure_1/frame_1"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr(f"{frame}/raw.png", png(8, 4))
                target.writestr(f"{frame}/instrument_instances.png", png(8, 4, 1))
                target.writestr(
                    f"{frame}/toolposes.json",
                    json.dumps({"instances": [{"nodes": [[1, 2]], "tags": [2]}]}),
                )
            report = audit(archive, root / "output", expected_frames=1)
            self.assertEqual(report["detected_frame_directories"], 1)
            self.assertEqual(report["complete_triplets"], 1)
            self.assertTrue(report["frame_count_matches_expectation"])
            self.assertEqual(report["mismatched_image_mask_dimensions"], 0)
            self.assertEqual(report["split_counts"], {"training": 1})


if __name__ == "__main__":
    unittest.main()
