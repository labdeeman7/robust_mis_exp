#!/usr/bin/env python3
"""Score and render the six manually reviewed bucket-C correspondences."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from build_c2_manifests import read_grayscale_png, valid_node
from render_c2_examples import (
    MASK_COLORS, clone, decode_rgb_png, draw_disk, draw_line, encode_rgb_png,
    frame_dir, paste, set_pixel, tint_masks,
)


def centreline_pixels(pose: dict[str, Any], width: int, height: int) -> set[tuple[int, int]]:
    nodes = pose.get("nodes", [])
    pixels: set[tuple[int, int]] = set()
    for edge in pose.get("edges", []):
        if not (isinstance(edge, list) and len(edge) == 2 and max(edge) < len(nodes)):
            continue
        start, end = nodes[edge[0]], nodes[edge[1]]
        if not (valid_node(start) and valid_node(end)):
            continue
        steps = max(1, math.ceil(max(abs(end[0] - start[0]), abs(end[1] - start[1]))))
        for step in range(steps + 1):
            fraction = step / steps
            x = round(start[0] + (end[0] - start[0]) * fraction)
            y = round(start[1] + (end[1] - start[1]) * fraction)
            if 0 <= x < width and 0 <= y < height:
                pixels.add((x, y))
    return pixels


def coverage(mask_rows: list[bytearray], pixels: set[tuple[int, int]], label: int, radius: int) -> float:
    """Fraction of in-frame skeleton centreline within radius pixels of label."""
    if not pixels:
        return 0.0
    height, width = len(mask_rows), len(mask_rows[0])
    offsets = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
               if dx * dx + dy * dy <= radius * radius]
    covered = 0
    for x, y in pixels:
        if any(0 <= x + dx < width and 0 <= y + dy < height and mask_rows[y + dy][x + dx] == label
               for dx, dy in offsets):
            covered += 1
    return covered / len(pixels)


def best_assignment(scores: list[dict[int, float]], labels: list[int]) -> list[int]:
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row in scores:
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (total, assignment) in states.items():
            for index, label in enumerate(labels):
                if used & (1 << index):
                    continue
                candidate = (total + row[label], assignment + (label,))
                key = used | (1 << index)
                if key not in updated or candidate[0] > updated[key][0]:
                    updated[key] = candidate
        states = updated
    return list(max(states.values(), key=lambda item: item[0])[1])


def draw_poses(rows: list[bytearray], width: int, height: int,
               poses: list[dict[str, Any]], assignment: list[int]) -> None:
    for pose, label in zip(poses, assignment):
        color = MASK_COLORS.get(label, (255, 255, 255))
        nodes = pose.get("nodes", [])
        for edge in pose.get("edges", []):
            if isinstance(edge, list) and len(edge) == 2 and max(edge) < len(nodes):
                start, end = nodes[edge[0]], nodes[edge[1]]
                if valid_node(start) and valid_node(end):
                    draw_line(rows, width, height, start, end, color)
        for node, tag in zip(nodes, pose.get("tags", [])):
            if valid_node(node):
                draw_disk(rows, width, height, node[0], node[1], 7 if tag == "visible" else 5, color)


def render_case(source: Path, destination: Path, manual: list[int], radius: int):
    width, height, raw = decode_rgb_png(source / "raw.png")
    mask = read_grayscale_png(source / "instrument_instances.png")
    poses: list[dict[str, Any]] = json.loads((source / "toolposes.json").read_text())
    labels = sorted({value for row in mask.rows for value in row if value})
    scores = [{label: coverage(mask.rows, centreline_pixels(pose, width, height), label, radius)
               for label in labels} for pose in poses]
    automatic = best_assignment(scores, labels)
    panels = [clone(raw) for _ in range(4)]
    tint_masks(panels[1], mask.rows, alpha=0.62)
    draw_poses(panels[2], width, height, poses, automatic)
    draw_poses(panels[3], width, height, poses, manual)
    separator = 6
    combined_width = width * 4 + separator * 3
    combined = [bytearray(combined_width * 3) for _ in range(height)]
    for index, panel in enumerate(panels):
        paste(combined, panel, index * (width + separator))
    for boundary in range(1, 4):
        start = boundary * width + (boundary - 1) * separator
        for y in range(height):
            for x in range(start, start + separator):
                set_pixel(combined, combined_width, height, x, y, (255, 255, 255))
    encode_rgb_png(destination, combined_width, height, combined)
    return automatic, scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/RobustMIPS"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/c2/bucket_c_manual_resolved.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/c2_bucket_c_matched"))
    parser.add_argument("--dilation-radius", type=int, default=20)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    cases: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        cases.setdefault(int(row["case_id"]), []).append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_rows = []
    readme = ["# Bucket C: explicit colour-matched review", "",
              "Panel order: **original | masks | 20 px dilation matcher | manual review**.", "",
              "Mask and skeleton colours now have one meaning: cyan = label 50; pink = label 100.",
              "The third panel colours poses by the automatic assignment; the fourth uses your reviewed assignment.", ""]
    for case_id, case_rows in sorted(cases.items()):
        case_rows.sort(key=lambda row: int(row["pose_index"]))
        key = case_rows[0]["frame_key"]
        manual = [int(row["final_mask_label"]) for row in case_rows]
        filename = f"{case_id:02d}_matched.png"
        automatic, scores = render_case(frame_dir(args.root, key), args.output_dir / filename,
                                        manual, args.dilation_radius)
        readme += [f"## {case_id:02d} — `{key}`", "", f"![case {case_id}]({filename})", "",
                   f"Automatic: `{automatic}`; manual: `{manual}`.", ""]
        for pose_index, row_scores in enumerate(scores):
            score_rows.append({"case_id": case_id, "frame_key": key, "pose_index": pose_index,
                               "coverage_mask_50": f"{row_scores.get(50, 0.0):.6f}",
                               "coverage_mask_100": f"{row_scores.get(100, 0.0):.6f}",
                               "automatic_mask_label": automatic[pose_index],
                               "manual_mask_label": manual[pose_index],
                               "automatic_matches_manual": automatic[pose_index] == manual[pose_index]})
    fields = list(score_rows[0])
    with (args.output_dir / "dilated_coverage_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(score_rows)
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Rendered {len(cases)} cases to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
