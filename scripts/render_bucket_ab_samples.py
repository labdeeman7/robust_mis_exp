#!/usr/bin/env python3
"""Render deterministic random samples from correspondence buckets A and B."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from build_c2_manifests import read_grayscale_png
from render_bucket_c_matches import best_assignment, centreline_pixels, coverage, draw_poses
from render_c2_examples import (
    clone, decode_rgb_png, encode_rgb_png, frame_dir, paste, set_pixel, tint_masks,
)


def truth(value: str) -> bool:
    return value.lower() == "true"


def choose(rows: list[dict[str, str]], off_image: bool, count: int, seed: int) -> list[dict[str, str]]:
    eligible = [row for row in rows
                if truth(row["counts_match"])
                and truth(row["canonical_mask_label_set"])
                and truth(row["has_off_image_keypoint"]) == off_image
                and int(row["pose_instance_count"]) > 0]
    singles = [row for row in eligible if int(row["pose_instance_count"]) == 1]
    multiples = [row for row in eligible if int(row["pose_instance_count"]) >= 2]
    rng = random.Random(seed)
    rng.shuffle(singles)
    rng.shuffle(multiples)
    # Correspondence is easiest to audit with multiple instruments, while single
    # instruments remain important for the intended SAM evaluation population.
    single_count = count // 2
    selected = singles[:single_count] + multiples[:count - single_count]
    rng.shuffle(selected)
    if len(selected) != count:
        raise RuntimeError(f"only found {len(selected)} eligible frames")
    return selected


def render(source: Path, destination: Path, radius: int):
    width, height, raw = decode_rgb_png(source / "raw.png")
    mask = read_grayscale_png(source / "instrument_instances.png")
    poses = json.loads((source / "toolposes.json").read_text())
    labels = sorted({value for row in mask.rows for value in row if value})
    scores = [{label: coverage(mask.rows, centreline_pixels(pose, width, height), label, radius)
               for label in labels} for pose in poses]
    assignment = best_assignment(scores, labels)
    panels = [clone(raw) for _ in range(3)]
    tint_masks(panels[1], mask.rows, alpha=0.62)
    draw_poses(panels[2], width, height, poses, assignment)
    separator = 6
    combined_width = width * 3 + separator * 2
    combined = [bytearray(combined_width * 3) for _ in range(height)]
    for index, panel in enumerate(panels):
        paste(combined, panel, index * (width + separator))
    for boundary in range(1, 3):
        start = boundary * width + (boundary - 1) * separator
        for y in range(height):
            for x in range(start, start + separator):
                set_pixel(combined, combined_width, height, x, y, (255, 255, 255))
    encode_rgb_png(destination, combined_width, height, combined)
    return assignment, scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("data/manifests/c2/frames.csv"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/RobustMIPS"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--dilation-radius", type=int, default=20)
    args = parser.parse_args()
    with args.frames.open(newline="", encoding="utf-8") as stream:
        frames = list(csv.DictReader(stream))
    for bucket, off_image, seed in (("a", False, 4201), ("b", True, 4202)):
        output = args.output_root / f"c2_bucket_{bucket}_matched"
        output.mkdir(parents=True, exist_ok=True)
        selected = choose(frames, off_image, args.count, seed)
        records = []
        readme = [f"# Bucket {bucket.upper()} matched examples", "",
                  "Panel order: **original | coloured instance masks | colour-matched poses**.", "",
                  "A pose and its assigned mask always share the same colour. Assignment maximizes one-to-one",
                  f"skeleton coverage within a {args.dilation_radius}-pixel mask dilation.", ""]
        for number, row in enumerate(selected, 1):
            filename = f"{number:02d}_{row['split']}_{row['pose_instance_count']}poses.png"
            assignment, scores = render(frame_dir(args.root, row["frame_key"]), output / filename,
                                        args.dilation_radius)
            records.append({"sample": number, "bucket": bucket.upper(), "frame_key": row["frame_key"],
                            "pose_count": row["pose_instance_count"], "mask_count": row["mask_instance_count"],
                            "has_off_image_keypoint": row["has_off_image_keypoint"],
                            "assignment_by_pose_index": json.dumps(assignment),
                            "coverage_scores": json.dumps(scores, sort_keys=True), "image": filename})
            readme += [f"## {number:02d} — `{row['frame_key']}`", "", f"![sample {number}]({filename})", "",
                       f"Pose-index assignment: `{assignment}`.", ""]
        with (output / "selection_and_scores.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        print(f"Rendered {len(records)} bucket {bucket.upper()} frames to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
