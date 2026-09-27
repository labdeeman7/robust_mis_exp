#!/usr/bin/env python3
"""Render selected C2 correspondence examples as standalone PNG panels."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

from build_c2_manifests import read_grayscale_png, valid_node


MASK_COLORS = {
    50: (0, 229, 255),
    100: (255, 61, 113),
    150: (124, 255, 0),
    200: (255, 214, 0),
    250: (199, 125, 255),
    44: (255, 140, 66),
    94: (0, 245, 160),
}
POSE_COLORS = ((0, 229, 255), (255, 61, 113), (124, 255, 0), (255, 214, 0), (199, 125, 255))


def decode_rgb_png(path: Path) -> tuple[int, int, list[bytearray]]:
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG: {path}")
    position = 8
    compressed: list[bytes] = []
    width = height = bit_depth = color_type = interlace = None
    while position < len(payload):
        length = struct.unpack(">I", payload[position : position + 4])[0]
        kind = payload[position + 4 : position + 8]
        data = payload[position + 8 : position + 8 + length]
        position += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", data)
        elif kind == b"IDAT":
            compressed.append(data)
        elif kind == b"IEND":
            break
    if bit_depth != 8 or color_type != 2 or interlace != 0:
        raise ValueError(f"expected non-interlaced RGB8 PNG: {path}")
    raw = zlib.decompress(b"".join(compressed))
    stride, bpp = int(width) * 3, 3
    previous = bytearray(stride)
    rows: list[bytearray] = []
    offset = 0
    for _ in range(int(height)):
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + stride])
        offset += stride
        for x in range(stride):
            left = row[x - bpp] if x >= bpp else 0
            above = previous[x]
            upper_left = previous[x - bpp] if x >= bpp else 0
            if filter_type == 1:
                row[x] = (row[x] + left) & 255
            elif filter_type == 2:
                row[x] = (row[x] + above) & 255
            elif filter_type == 3:
                row[x] = (row[x] + ((left + above) // 2)) & 255
            elif filter_type == 4:
                estimate = left + above - upper_left
                dl, da, du = abs(estimate - left), abs(estimate - above), abs(estimate - upper_left)
                predictor = left if dl <= da and dl <= du else above if da <= du else upper_left
                row[x] = (row[x] + predictor) & 255
            elif filter_type != 0:
                raise ValueError(f"unsupported filter {filter_type}")
        rows.append(row)
        previous = row
    return int(width), int(height), rows


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode_rgb_png(path: Path, width: int, height: int, rows: list[bytearray]) -> None:
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(raw, 6))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def clone(rows: list[bytearray]) -> list[bytearray]:
    return [bytearray(row) for row in rows]


def set_pixel(rows: list[bytearray], width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if not (0 <= x < width and 0 <= y < height):
        return
    offset = x * 3
    rows[y][offset : offset + 3] = bytes(color)


def draw_disk(
    rows: list[bytearray], width: int, height: int, cx: float, cy: float, radius: int, color: tuple[int, int, int]
) -> None:
    icx, icy = round(cx), round(cy)
    for y in range(icy - radius, icy + radius + 1):
        for x in range(icx - radius, icx + radius + 1):
            if (x - icx) ** 2 + (y - icy) ** 2 <= radius**2:
                set_pixel(rows, width, height, x, y, color)


def draw_line(
    rows: list[bytearray], width: int, height: int, start: list[float], end: list[float], color: tuple[int, int, int]
) -> None:
    distance = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    steps = max(1, math.ceil(distance))
    for step in range(steps + 1):
        fraction = step / steps
        x = start[0] + (end[0] - start[0]) * fraction
        y = start[1] + (end[1] - start[1]) * fraction
        draw_disk(rows, width, height, x, y, 2, color)


def tint_masks(rows: list[bytearray], mask_rows: list[bytearray], alpha: float = 0.52) -> None:
    for y, labels in enumerate(mask_rows):
        row = rows[y]
        for x, label in enumerate(labels):
            if label == 0:
                continue
            color = MASK_COLORS.get(label, (255, 255, 255))
            offset = x * 3
            for channel in range(3):
                row[offset + channel] = round((1 - alpha) * row[offset + channel] + alpha * color[channel])


def paste(destination: list[bytearray], source: list[bytearray], x_offset: int) -> None:
    for target_row, source_row in zip(destination, source):
        start = x_offset * 3
        target_row[start : start + len(source_row)] = source_row


def render(frame_dir: Path, instrument_rows: list[dict[str, str]], destination: Path) -> None:
    width, height, raw = decode_rgb_png(frame_dir / "raw.png")
    mask = read_grayscale_png(frame_dir / "instrument_instances.png")
    poses: list[dict[str, Any]] = json.loads((frame_dir / "toolposes.json").read_text())
    original = clone(raw)
    mask_panel = clone(raw)
    pose_panel = clone(raw)
    tint_masks(mask_panel, mask.rows)
    for pose_index, pose in enumerate(poses):
        color = POSE_COLORS[pose_index % len(POSE_COLORS)]
        nodes = pose.get("nodes", [])
        for edge in pose.get("edges", []):
            if len(edge) != 2 or max(edge) >= len(nodes):
                continue
            a, b = nodes[edge[0]], nodes[edge[1]]
            if valid_node(a) and valid_node(b):
                draw_line(pose_panel, width, height, a, b, color)
        for node, tag in zip(nodes, pose.get("tags", [])):
            if valid_node(node):
                draw_disk(pose_panel, width, height, node[0], node[1], 7 if tag == "visible" else 5, color)
    separator = 6
    combined_width = width * 3 + separator * 2
    combined = [bytearray(combined_width * 3) for _ in range(height)]
    paste(combined, original, 0)
    paste(combined, mask_panel, width + separator)
    paste(combined, pose_panel, width * 2 + separator * 2)
    for y in range(height):
        for x in range(width, width + separator):
            set_pixel(combined, combined_width, height, x, y, (255, 255, 255))
        for x in range(width * 2 + separator, width * 2 + separator * 2):
            set_pixel(combined, combined_width, height, x, y, (255, 255, 255))
    encode_rgb_png(destination, combined_width, height, combined)


EXAMPLES = {
    "ambiguous_1_off_image": "Stage_1/Proctocolectomy/1/141000",
    "ambiguous_2_pose_without_mask": "Stage_3/Sigmoid/10/104928",
    "ambiguous_3_crossing_low_margin": "Stage_2/Proctocolectomy/6/190500",
    "ambiguous_4_single_visible_point": "Stage_1/Proctocolectomy/10/33000",
    "count_mismatch_zero_pose_two_masks": "Training/Proctocolectomy/10/35310",
}


def frame_dir(root: Path, key: str) -> Path:
    split, surgery, procedure, frame = key.split("/")
    if split == "Training":
        return root / "Training" / surgery / procedure / frame
    return root / "Testing" / split / surgery / procedure / frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/RobustMIPS"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/c2/instruments.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/c2_requested_examples"))
    args = parser.parse_args()
    by_frame: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    with args.manifest.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            by_frame[row["frame_key"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, key in EXAMPLES.items():
        destination = args.output_dir / f"{name}.png"
        render(frame_dir(args.root, key), by_frame.get(key, []), destination)
        records.append({"name": name, "frame_key": key, "file": destination.name})
        print(destination)
    (args.output_dir / "examples.json").write_text(json.dumps(records, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
