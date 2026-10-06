#!/usr/bin/env python3
"""Build deterministic SAM 1 prompts for every C4 pilot instrument."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_c2_manifests import valid_node


KEYPOINT_NAMES = ("entry", "hinge", "tip1", "tip2")
STRATEGIES = (
    "p1_visible_points",
    "p2_visible_skeleton",
    "p3_visible_box",
    "p4_visible_points_box",
    "p5_visible_points_negatives",
    "p6_visible_points_box_negatives",
    "p7_shifted_entry_ablation",
)

BUCKET_ORDER = ("A", "B", "C", "D")
BUCKET_DESCRIPTIONS = {
    "A": "complete: all keypoints visible and in bounds",
    "B": "partial in-frame: at least one occluded or missing keypoint, none off-screen",
    "C": "off-screen: at least one keypoint coordinate outside the image",
    "D": "correspondence-hard: pose-to-mask assignment required explicit review",
}


def is_in_bounds(point: list[float], width: int, height: int) -> bool:
    return valid_node(point) and 0 <= point[0] < width and 0 <= point[1] < height


def pose_bucket(pose: dict[str, Any], width: int, height: int, correspondence_hard: bool = False) -> str:
    """Assign the mutually exclusive reporting stratum for one instrument pose."""
    if correspondence_hard:
        return "D"
    nodes = pose.get("nodes", [])
    if any(valid_node(point) and not is_in_bounds(point, width, height) for point in nodes):
        return "C"
    tags = pose.get("tags", [])
    if len(tags) != len(nodes) or any(tag != "visible" for tag in tags):
        return "B"
    return "A"


def visible_points(pose: dict[str, Any], width: int, height: int) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point, tag in zip(pose.get("nodes", []), pose.get("tags", []))
            if tag == "visible" and is_in_bounds(point, width, height)]


def visible_skeleton(pose: dict[str, Any], width: int, height: int, spacing: float = 20.0) -> list[list[float]]:
    nodes, tags = pose.get("nodes", []), pose.get("tags", [])
    result = visible_points(pose, width, height)
    for edge in pose.get("edges", []):
        if not (isinstance(edge, list) and len(edge) == 2 and max(edge) < len(nodes)):
            continue
        a, b = edge
        if tags[a] != "visible" or tags[b] != "visible" or not is_in_bounds(nodes[a], width, height) or not is_in_bounds(nodes[b], width, height):
            continue
        distance = math.dist(nodes[a], nodes[b])
        steps = max(1, math.ceil(distance / spacing))
        for step in range(1, steps):
            fraction = step / steps
            result.append([nodes[a][0] + (nodes[b][0] - nodes[a][0]) * fraction,
                           nodes[a][1] + (nodes[b][1] - nodes[a][1]) * fraction])
    return deduplicate(result)


def deduplicate(points: list[list[float]]) -> list[list[float]]:
    seen: set[tuple[float, float]] = set()
    result = []
    for x, y in points:
        key = (round(float(x), 3), round(float(y), 3))
        if key not in seen:
            seen.add(key)
            result.append([float(x), float(y)])
    return result


def prompt_box(points: list[list[float]], width: int, height: int, padding: float = 20.0) -> list[float] | None:
    if not points:
        return None
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    return [max(0.0, min(xs) - padding), max(0.0, min(ys) - padding),
            min(width - 1.0, max(xs) + padding), min(height - 1.0, max(ys) + padding)]


def shifted_entry(pose: dict[str, Any], width: int, height: int, inward: float = 10.0) -> list[float] | None:
    """Project an out-of-bounds entry toward the nearest visible node, then move inward."""
    nodes, tags = pose.get("nodes", []), pose.get("tags", [])
    if not nodes or not valid_node(nodes[0]) or is_in_bounds(nodes[0], width, height):
        return None
    visible = [point for point, tag in zip(nodes[1:], tags[1:]) if tag == "visible" and is_in_bounds(point, width, height)]
    if not visible:
        return None
    entry = nodes[0]
    target = min(visible, key=lambda point: math.dist(entry, point))
    dx, dy = target[0] - entry[0], target[1] - entry[1]
    intersections = []
    for boundary, start, delta in ((0.0, entry[0], dx), (width - 1.0, entry[0], dx),
                                   (0.0, entry[1], dy), (height - 1.0, entry[1], dy)):
        if delta == 0:
            continue
        t = (boundary - start) / delta
        if 0 <= t <= 1:
            x, y = entry[0] + t * dx, entry[1] + t * dy
            if 0 <= x < width and 0 <= y < height:
                intersections.append((t, x, y))
    if not intersections:
        return None
    _, x, y = min(intersections)
    length = math.hypot(dx, dy)
    step = min(inward / length, 1.0)
    return [min(width - 1.0, max(0.0, x + dx * step)), min(height - 1.0, max(0.0, y + dy * step))]


def make_prompt(strategy: str, pose: dict[str, Any], other_poses: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    visible = visible_points(pose, width, height)
    positives = visible_skeleton(pose, width, height) if strategy == "p2_visible_skeleton" else list(visible)
    negatives: list[list[float]] = []
    box = None
    if strategy in {"p3_visible_box"}:
        positives = []
    if strategy in {"p3_visible_box", "p4_visible_points_box", "p6_visible_points_box_negatives"}:
        box = prompt_box(visible, width, height)
    if strategy in {"p5_visible_points_negatives", "p6_visible_points_box_negatives"}:
        negatives = deduplicate([point for other in other_poses for point in visible_points(other, width, height)])
    shifted = None
    if strategy == "p7_shifted_entry_ablation":
        shifted = shifted_entry(pose, width, height)
        if shifted is not None:
            positives = deduplicate(positives + [shifted])
    return {"positive_points": positives, "negative_points": negatives, "box_xyxy": box,
            "shifted_entry_point": shifted}


def main() -> int:
    pilot_dir = Path("data/manifests/c4_pilot")
    output = Path("data/manifests/c5_prompts")
    with (pilot_dir / "frames.csv").open(newline="", encoding="utf-8") as stream:
        frames = {row["frame_key"]: row for row in csv.DictReader(stream)}
    with (pilot_dir / "instances.csv").open(newline="", encoding="utf-8") as stream:
        instances = list(csv.DictReader(stream))
    poses_by_frame: dict[str, list[dict[str, Any]]] = {}
    output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with (output / "prompts.jsonl").open("w", encoding="utf-8") as destination:
        for instance in instances:
            key = instance["frame_key"]
            frame = frames[key]
            poses = poses_by_frame.setdefault(key, json.loads(Path(instance["pose_path"]).read_text()))
            pose_index = int(instance["pose_index"])
            pose = poses[pose_index]
            other_poses = [other for index, other in enumerate(poses) if index != pose_index]
            for strategy in STRATEGIES:
                prompt = make_prompt(strategy, pose, other_poses, int(frame["width"]), int(frame["height"]))
                row = {"prompt_id": f"{instance['pilot_index']}:{pose_index}:{strategy}",
                       "pilot_index": int(instance["pilot_index"]), "bucket": instance["bucket"],
                       "frame_key": key, "pose_index": pose_index, "mask_label": int(instance["mask_label"]),
                       "image_path": instance["image_path"], "mask_path": instance["mask_path"],
                       "strategy": strategy, **prompt}
                destination.write(json.dumps(row, separators=(",", ":")) + "\n")
                counts[strategy] += 1
                counts[f"{strategy}_no_positive"] += not bool(prompt["positive_points"])
                counts[f"{strategy}_shifted"] += prompt["shifted_entry_point"] is not None
    summary = {"instance_count": len(instances), "strategy_count": len(STRATEGIES),
               "prompt_count": len(instances) * len(STRATEGIES), "strategies": STRATEGIES,
               "counts": counts}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
