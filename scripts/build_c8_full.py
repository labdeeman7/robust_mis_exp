#!/usr/bin/env python3
"""Build the full, non-overlapping A/B/C instance manifest for C8."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_c2_manifests import read_grayscale_png
from render_bucket_c_matches import best_assignment, centreline_pixels, coverage
from build_c5_prompts import BUCKET_DESCRIPTIONS, BUCKET_ORDER, pose_bucket


def truth(value: str) -> bool:
    return value.lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    c2 = Path("data/manifests/c2")
    output = Path("data/manifests/c8_full")
    frames = read_csv(c2 / "frames.csv")
    easy = read_csv(c2 / "easy_matched_instruments.csv")
    reviewed = read_csv(c2 / "bucket_c_manual_resolved.csv")
    easy_by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    reviewed_by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in easy:
        easy_by_frame[row["frame_key"]].append(row)
    for row in reviewed:
        reviewed_by_frame[row["frame_key"]].append(row)
    c_keys = set(reviewed_by_frame)
    frame_rows, instance_rows = [], []
    recomputed_frames = 0
    for frame in frames:
        pose_count = int(frame["pose_instance_count"])
        if pose_count == 0 or not truth(frame["counts_match"]):
            continue
        key = frame["frame_key"]
        correspondence_hard = key in c_keys
        if correspondence_hard:
            mappings = [(int(row["pose_index"]), int(row["final_mask_label"]), "manual_review")
                        for row in sorted(reviewed_by_frame[key], key=lambda row: int(row["pose_index"]))]
        else:
            easy_rows = sorted(easy_by_frame.get(key, []), key=lambda row: int(row["instance_index"]))
            if len(easy_rows) == pose_count:
                mappings = [(int(row["instance_index"]), int(row["matched_mask_label"]), "c2_geometric")
                            for row in easy_rows]
            else:
                poses = json.loads(Path(frame["pose_path"]).read_text())
                mask = read_grayscale_png(Path(frame["mask_path"]))
                labels = sorted({value for row in mask.rows for value in row if value})
                scores = [{label: coverage(mask.rows, centreline_pixels(pose, mask.width, mask.height), label, 20)
                           for label in labels} for pose in poses]
                assignment = best_assignment(scores, labels)
                mappings = [(index, label, "dilated_skeleton_20px") for index, label in enumerate(assignment)]
                recomputed_frames += 1
        if len(mappings) != pose_count or any(label is None for _, label, _ in mappings):
            raise RuntimeError(f"incomplete mapping for {key}: {mappings}")
        evaluation_frame_index = len(frame_rows)
        poses = json.loads(Path(frame["pose_path"]).read_text())
        instance_buckets = [pose_bucket(poses[pose_index], int(frame["width"]), int(frame["height"]),
                                        correspondence_hard) for pose_index, _, _ in mappings]
        frame_bucket = max(instance_buckets, key=BUCKET_ORDER.index)
        frame_rows.append({"evaluation_frame_index": evaluation_frame_index, "bucket": frame_bucket, **frame})
        for (pose_index, mask_label, source), bucket in zip(mappings, instance_buckets):
            instance_rows.append({"evaluation_instance_index": len(instance_rows),
                                  "evaluation_frame_index": evaluation_frame_index, "bucket": bucket,
                                  "frame_key": key, "pose_index": pose_index, "mask_label": mask_label,
                                  "match_source": source, "image_path": frame["image_path"],
                                  "mask_path": frame["mask_path"], "pose_path": frame["pose_path"]})
    if len(frame_rows) != 7348 or len(instance_rows) != 10525:
        raise AssertionError(f"unexpected full-set size: {len(frame_rows)} frames, {len(instance_rows)} instances")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("frames.csv", frame_rows), ("instances.csv", instance_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    summary = {"frame_count": len(frame_rows), "instance_count": len(instance_rows),
               "bucket_definitions": BUCKET_DESCRIPTIONS,
               "bucket_frames": Counter(row["bucket"] for row in frame_rows),
               "bucket_instances": Counter(row["bucket"] for row in instance_rows),
               "match_sources": Counter(row["match_source"] for row in instance_rows),
               "recomputed_frame_count": recomputed_frames}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
