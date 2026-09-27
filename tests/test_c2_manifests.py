from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_c2_manifests import (  # noqa: E402
    mask_statistics,
    optimal_assignment,
    read_grayscale_png,
    sample_segment,
)


def grayscale_png(rows: list[bytes]) -> bytes:
    height, width = len(rows), len(rows[0])

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    pixels = b"".join(b"\x00" + row for row in rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


class C2ManifestTest(unittest.TestCase):
    def test_png_decode_and_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            path.write_bytes(grayscale_png([bytes([0, 50, 50]), bytes([0, 0, 100])]))
            mask = read_grayscale_png(path)
            self.assertEqual((mask.width, mask.height), (3, 2))
            stats = mask_statistics(mask)
            self.assertEqual(stats[50]["area"], 2)
            self.assertEqual(stats[50]["bbox_width"], 2)
            self.assertEqual(stats[100]["centroid_x"], 2)

    def test_segment_sampling_includes_endpoints(self) -> None:
        points = list(sample_segment([0.0, 0.0], [4.0, 0.0], spacing=2.0))
        self.assertEqual(points, [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)])

    def test_assignment_handles_swapped_pose_order(self) -> None:
        scores = [{50: 0.0, 100: 0.9}, {50: 0.8, 100: 0.1}]
        self.assertEqual(optimal_assignment(scores, [50, 100]), [100, 50])

    def test_assignment_can_leave_pose_unmatched(self) -> None:
        self.assertEqual(optimal_assignment([{50: 0.0}], [50]), [None])


if __name__ == "__main__":
    unittest.main()
