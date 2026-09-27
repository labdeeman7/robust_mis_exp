#!/usr/bin/env python3
"""Build the deterministic 200-image C4 pilot across buckets A, B, and C."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260927
QUOTAS = {"A": (50, 50), "B": (47, 47)}  # (single-instrument, multi-instrument)


def truth(value: str) -> bool:
    return value.lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    manifest_dir = Path("data/manifests/c2")
    output = Path("data/manifests/c4_pilot")
    frames = read_csv(manifest_dir / "frames.csv")
    easy = read_csv(manifest_dir / "easy_matched_instruments.csv")
    reviewed = read_csv(manifest_dir / "bucket_c_manual_resolved.csv")
    frame_by_key = {row["frame_key"]: row for row in frames}
    easy_by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in easy:
        easy_by_frame[row["frame_key"]].append(row)
    c_keys = {row["frame_key"] for row in reviewed}
    reviewed_by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reviewed:
        reviewed_by_frame[row["frame_key"]].append(row)

    candidates: dict[str, list[dict[str, str]]] = {"A": [], "B": []}
    for row in frames:
        key = row["frame_key"]
        pose_count = int(row["pose_instance_count"])
        if key in c_keys or pose_count == 0 or not truth(row["counts_match"]):
            continue
        matched = easy_by_frame.get(key, [])
        if len(matched) != pose_count:
            continue
        bucket = "B" if truth(row["has_off_image_keypoint"]) else "A"
        candidates[bucket].append(row)

    rng = random.Random(SEED)
    selected: list[tuple[str, dict[str, str]]] = []
    for bucket, (single_quota, multi_quota) in QUOTAS.items():
        singles = [row for row in candidates[bucket] if int(row["pose_instance_count"]) == 1]
        multiples = [row for row in candidates[bucket] if int(row["pose_instance_count"]) >= 2]
        rng.shuffle(singles)
        rng.shuffle(multiples)
        if len(singles) < single_quota or len(multiples) < multi_quota:
            raise RuntimeError(f"insufficient {bucket} candidates")
        selected.extend((bucket, row) for row in singles[:single_quota] + multiples[:multi_quota])
    selected.extend(("C", frame_by_key[key]) for key in sorted(c_keys))
    rng.shuffle(selected)

    frame_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    for pilot_index, (bucket, frame) in enumerate(selected):
        key = frame["frame_key"]
        frame_rows.append({"pilot_index": pilot_index, "bucket": bucket, **frame})
        if bucket == "C":
            matches = sorted(reviewed_by_frame[key], key=lambda row: int(row["pose_index"]))
            mappings = [(int(row["pose_index"]), int(row["final_mask_label"]), "manual_review") for row in matches]
        else:
            matches = sorted(easy_by_frame[key], key=lambda row: int(row["instance_index"]))
            mappings = [(int(row["instance_index"]), int(row["matched_mask_label"]), "c2_geometric")
                        for row in matches]
        for pose_index, mask_label, source in mappings:
            instance_rows.append({"pilot_index": pilot_index, "bucket": bucket, "frame_key": key,
                                  "pose_index": pose_index, "mask_label": mask_label,
                                  "match_source": source, "image_path": frame["image_path"],
                                  "mask_path": frame["mask_path"], "pose_path": frame["pose_path"]})

    if len(frame_rows) != 200 or len({row["frame_key"] for row in frame_rows}) != 200:
        raise AssertionError("pilot must contain 200 distinct frames")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("frames.csv", frame_rows), ("instances.csv", instance_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "seed": SEED,
        "frame_count": len(frame_rows),
        "instance_count": len(instance_rows),
        "bucket_frames": Counter(row["bucket"] for row in frame_rows),
        "bucket_instances": Counter(row["bucket"] for row in instance_rows),
        "single_vs_multi_frames": Counter(
            f"{row['bucket']}_{'single' if int(row['pose_instance_count']) == 1 else 'multi'}"
            for row in frame_rows
        ),
        "split_frames": Counter(row["split"] for row in frame_rows),
        "surgery_frames": Counter(row["surgery_type"] for row in frame_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
