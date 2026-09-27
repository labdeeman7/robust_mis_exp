#!/usr/bin/env python3
"""Generate only P1 and P2 prompts for the full C8 evaluation."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from build_c5_prompts import make_prompt


STRATEGIES = ("p1_visible_points", "p2_visible_skeleton")


def main() -> int:
    root = Path("data/manifests/c8_full")
    with (root / "frames.csv").open(newline="", encoding="utf-8") as stream:
        frames = {row["frame_key"]: row for row in csv.DictReader(stream)}
    with (root / "instances.csv").open(newline="", encoding="utf-8") as stream:
        instances = list(csv.DictReader(stream))
    pose_cache = {}
    counts = Counter()
    destination = root / "prompts_p1_p2.jsonl"
    with destination.open("w", encoding="utf-8") as stream:
        for instance in instances:
            frame = frames[instance["frame_key"]]
            poses = pose_cache.setdefault(instance["frame_key"], json.loads(Path(instance["pose_path"]).read_text()))
            pose_index = int(instance["pose_index"])
            for strategy in STRATEGIES:
                prompt = make_prompt(strategy, poses[pose_index], [], int(frame["width"]), int(frame["height"]))
                row = {"prompt_id": f"{instance['evaluation_instance_index']}:{pose_index}:{strategy}",
                       "pilot_index": int(instance["evaluation_instance_index"]),
                       "bucket": instance["bucket"], "frame_key": instance["frame_key"],
                       "pose_index": pose_index, "mask_label": int(instance["mask_label"]),
                       "image_path": instance["image_path"], "mask_path": instance["mask_path"],
                       "strategy": strategy, **prompt}
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                counts[strategy] += 1
                counts[f"{strategy}_no_positive"] += not bool(prompt["positive_points"])
    summary = {"frame_count": len(frames), "instance_count": len(instances),
               "strategy_count": len(STRATEGIES), "prompt_count": len(instances) * len(STRATEGIES),
               "strategies": STRATEGIES, "counts": counts}
    (root / "prompt_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
