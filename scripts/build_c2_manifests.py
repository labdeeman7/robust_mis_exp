#!/usr/bin/env python3
"""Build analysis-ready ROBUST-MIPS frame and instrument manifests.

This intentionally uses only the Python standard library so it can run on the
login node. Source files under data/raw are never modified.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


KEYPOINT_NAMES = ("entry", "hinge", "tip1", "tip2")
VALID_TAGS = {"visible", "occluded", "missing"}


@dataclass
class GrayImage:
    width: int
    height: int
    rows: list[bytearray]

    def value(self, x: float, y: float) -> int | None:
        ix, iy = round(x), round(y)
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return self.rows[iy][ix]
        return None


def read_grayscale_png(path: Path) -> GrayImage:
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature")
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
    if None in (width, height, bit_depth, color_type, interlace):
        raise ValueError("missing PNG IHDR")
    if bit_depth != 8 or color_type != 0 or interlace != 0:
        raise ValueError(
            f"expected non-interlaced 8-bit grayscale PNG; got depth={bit_depth}, "
            f"color_type={color_type}, interlace={interlace}"
        )
    raw = zlib.decompress(b"".join(compressed))
    stride = int(width)
    expected = int(height) * (stride + 1)
    if len(raw) != expected:
        raise ValueError(f"unexpected decoded size {len(raw)} (expected {expected})")
    previous = bytearray(stride)
    rows: list[bytearray] = []
    offset = 0
    for _ in range(int(height)):
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + stride])
        offset += stride
        for x in range(stride):
            left = row[x - 1] if x else 0
            above = previous[x]
            upper_left = previous[x - 1] if x else 0
            if filter_type == 1:
                row[x] = (row[x] + left) & 255
            elif filter_type == 2:
                row[x] = (row[x] + above) & 255
            elif filter_type == 3:
                row[x] = (row[x] + ((left + above) // 2)) & 255
            elif filter_type == 4:
                estimate = left + above - upper_left
                dl = abs(estimate - left)
                da = abs(estimate - above)
                du = abs(estimate - upper_left)
                predictor = left if dl <= da and dl <= du else above if da <= du else upper_left
                row[x] = (row[x] + predictor) & 255
            elif filter_type != 0:
                raise ValueError(f"unsupported PNG filter {filter_type}")
        rows.append(row)
        previous = row
    return GrayImage(int(width), int(height), rows)


def mask_statistics(mask: GrayImage) -> dict[int, dict[str, Any]]:
    stats: dict[int, list[int]] = {}
    # area, min_x, min_y, max_x, max_y, sum_x, sum_y
    for y, row in enumerate(mask.rows):
        for x, label in enumerate(row):
            if label == 0:
                continue
            if label not in stats:
                stats[label] = [0, x, y, x, y, 0, 0]
            value = stats[label]
            value[0] += 1
            value[1] = min(value[1], x)
            value[2] = min(value[2], y)
            value[3] = max(value[3], x)
            value[4] = max(value[4], y)
            value[5] += x
            value[6] += y
    return {
        label: {
            "area": value[0],
            "bbox_x": value[1],
            "bbox_y": value[2],
            "bbox_width": value[3] - value[1] + 1,
            "bbox_height": value[4] - value[2] + 1,
            "centroid_x": value[5] / value[0],
            "centroid_y": value[6] / value[0],
        }
        for label, value in stats.items()
    }


def valid_node(node: Any) -> bool:
    return (
        isinstance(node, list)
        and len(node) == 2
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in node)
    )


def sample_segment(start: list[float], end: list[float], spacing: float = 2.0) -> Iterator[tuple[float, float]]:
    distance = math.dist(start, end)
    count = max(1, math.ceil(distance / spacing))
    for step in range(count + 1):
        fraction = step / count
        yield (
            start[0] + (end[0] - start[0]) * fraction,
            start[1] + (end[1] - start[1]) * fraction,
        )


def skeleton_samples(pose: dict[str, Any], width: int, height: int) -> list[tuple[float, float]]:
    nodes = pose.get("nodes", [])
    tags = pose.get("tags", [])
    samples: list[tuple[float, float]] = []
    for node, tag in zip(nodes, tags):
        if valid_node(node) and tag == "visible" and 0 <= node[0] < width and 0 <= node[1] < height:
            samples.append((node[0], node[1]))
    for edge in pose.get("edges", []):
        if not (isinstance(edge, list) and len(edge) == 2 and all(isinstance(i, int) for i in edge)):
            continue
        a, b = edge
        if not (0 <= a < len(nodes) and 0 <= b < len(nodes)):
            continue
        if not (valid_node(nodes[a]) and valid_node(nodes[b])):
            continue
        if not (a < len(tags) and b < len(tags) and tags[a] == tags[b] == "visible"):
            continue
        samples.extend(
            (x, y)
            for x, y in sample_segment(nodes[a], nodes[b])
            if 0 <= x < width and 0 <= y < height
        )
    # Rounding can create many duplicates, so normalize at pixel resolution.
    return [(float(x), float(y)) for x, y in sorted({(round(x), round(y)) for x, y in samples})]


def overlap_scores(mask: GrayImage, samples: list[tuple[float, float]], labels: list[int]) -> dict[int, float]:
    if not samples:
        return {label: 0.0 for label in labels}
    counts = collections.Counter(mask.value(x, y) for x, y in samples)
    return {label: counts[label] / len(samples) for label in labels}


def optimal_assignment(score_rows: list[dict[int, float]], labels: list[int]) -> list[int | None]:
    """Maximum-overlap one-to-one pose/mask assignment, allowing unmatched poses."""
    if not score_rows or not labels:
        return [None] * len(score_rows)
    # Dynamic programming is tiny here (at most seven masks in release v1).
    states: dict[int, tuple[float, tuple[int | None, ...]]] = {0: (0.0, ())}
    for scores in score_rows:
        updated: dict[int, tuple[float, tuple[int | None, ...]]] = {}
        for used, (total, assignment) in states.items():
            candidates = [(used, total, assignment + (None,))]
            for label_index, label in enumerate(labels):
                if used & (1 << label_index):
                    continue
                score = scores.get(label, 0.0)
                # Zero-overlap links are not evidence of correspondence.
                if score <= 0:
                    continue
                candidates.append(
                    (used | (1 << label_index), total + score, assignment + (label,))
                )
            for new_used, new_total, new_assignment in candidates:
                previous = updated.get(new_used)
                if previous is None or new_total > previous[0]:
                    updated[new_used] = (new_total, new_assignment)
        states = updated
    _, best_assignment = max(states.values(), key=lambda item: item[0])
    return list(best_assignment)


def hierarchy(frame_dir: Path, root: Path) -> tuple[str, str, str, str]:
    parts = frame_dir.relative_to(root).parts
    if parts[0] == "Training" and len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    if parts[0] == "Testing" and len(parts) == 5:
        return parts[1], parts[2], parts[3], parts[4]
    raise ValueError(f"unexpected frame hierarchy: {frame_dir}")


def add_anomaly(
    anomalies: list[dict[str, Any]], frame_key: str, code: str, detail: str, instance_index: int | None = None
) -> None:
    anomalies.append(
        {"frame_key": frame_key, "instance_index": instance_index, "code": code, "detail": detail}
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(root: Path, output_dir: Path) -> dict[str, Any]:
    frame_rows: list[dict[str, Any]] = []
    instrument_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    tag_counts: collections.Counter[str] = collections.Counter()
    mask_label_counts: collections.Counter[int] = collections.Counter()
    pose_count_distribution: collections.Counter[int] = collections.Counter()
    split_counts: collections.Counter[str] = collections.Counter()
    off_image_by_keypoint: collections.Counter[str] = collections.Counter()
    missing_by_keypoint: collections.Counter[str] = collections.Counter()
    direct_order_testable = 0
    direct_order_supported = 0

    pose_paths = sorted(root.rglob("toolposes.json"))
    for number, pose_path in enumerate(pose_paths, start=1):
        frame_dir = pose_path.parent
        split, surgery_type, procedure_id, frame_id = hierarchy(frame_dir, root)
        frame_key = f"{split}/{surgery_type}/{procedure_id}/{frame_id}"
        image_path = frame_dir / "raw.png"
        mask_path = frame_dir / "instrument_instances.png"
        try:
            poses = json.loads(pose_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            add_anomaly(anomalies, frame_key, "invalid_pose_json", str(error))
            continue
        if not isinstance(poses, list):
            add_anomaly(anomalies, frame_key, "pose_top_level_not_list", type(poses).__name__)
            continue
        try:
            mask = read_grayscale_png(mask_path)
            stats = mask_statistics(mask)
        except (OSError, ValueError, zlib.error) as error:
            add_anomaly(anomalies, frame_key, "invalid_mask_png", str(error))
            continue
        labels = sorted(stats)
        expected_labels = [(50 * (index + 1)) % 256 for index in range(len(poses))]
        counts_match = len(poses) == len(labels)
        canonical_label_set = set(labels) == set(expected_labels)
        if not counts_match:
            add_anomaly(
                anomalies,
                frame_key,
                "pose_mask_count_mismatch",
                f"poses={len(poses)}, labels={labels}",
            )
        if counts_match and not canonical_label_set:
            add_anomaly(
                anomalies,
                frame_key,
                "unexpected_mask_labels",
                f"expected={expected_labels}, actual={labels}",
            )

        split_counts[split] += 1
        pose_count_distribution[len(poses)] += 1
        mask_label_counts.update(labels)
        frame_off_image = False
        frame_missing = False
        frame_identity_testable = True
        frame_identity_supported = True

        pose_analyses: list[dict[str, Any]] = []
        for index, pose in enumerate(poses):
            if not isinstance(pose, dict):
                add_anomaly(anomalies, frame_key, "pose_instance_not_object", type(pose).__name__, index)
                frame_identity_testable = False
                continue
            nodes = pose.get("nodes")
            tags = pose.get("tags")
            edges = pose.get("edges")
            transitions = pose.get("transitions")
            if not isinstance(nodes, list) or len(nodes) != 4:
                add_anomaly(anomalies, frame_key, "invalid_nodes", repr(nodes)[:300], index)
                nodes = nodes if isinstance(nodes, list) else []
            if not isinstance(tags, list) or len(tags) != 4:
                add_anomaly(anomalies, frame_key, "invalid_tags", repr(tags)[:300], index)
                tags = tags if isinstance(tags, list) else []
            if not isinstance(edges, list):
                add_anomaly(anomalies, frame_key, "invalid_edges", repr(edges)[:300], index)
            if not isinstance(transitions, list):
                add_anomaly(anomalies, frame_key, "invalid_transitions", repr(transitions)[:300], index)

            keypoint_values: dict[str, Any] = {}
            instance_off_image = False
            for keypoint_index, name in enumerate(KEYPOINT_NAMES):
                node = nodes[keypoint_index] if keypoint_index < len(nodes) else None
                tag = tags[keypoint_index] if keypoint_index < len(tags) else None
                tag_counts[str(tag)] += 1
                if tag not in VALID_TAGS:
                    add_anomaly(anomalies, frame_key, "invalid_visibility_tag", f"{name}={tag!r}", index)
                if tag == "missing":
                    missing_by_keypoint[name] += 1
                    frame_missing = True
                if node is not None and not valid_node(node):
                    add_anomaly(anomalies, frame_key, "invalid_coordinate", f"{name}={node!r}", index)
                x = node[0] if valid_node(node) else None
                y = node[1] if valid_node(node) else None
                off_image = bool(x is not None and not (0 <= x < mask.width and 0 <= y < mask.height))
                if off_image:
                    off_image_by_keypoint[name] += 1
                    instance_off_image = True
                    frame_off_image = True
                keypoint_values.update(
                    {
                        f"{name}_x": x,
                        f"{name}_y": y,
                        f"{name}_visibility": tag,
                        f"{name}_off_image": off_image,
                    }
                )

            samples = skeleton_samples(pose, mask.width, mask.height)
            scores = overlap_scores(mask, samples, labels)
            expected_label = (50 * (index + 1)) % 256
            expected_score = scores.get(expected_label, 0.0)
            best_score = max(scores.values(), default=0.0)
            best_labels = sorted(label for label, score in scores.items() if score == best_score and score > 0)
            identity_testable = bool(samples and best_score > 0 and expected_label in labels)
            identity_supported = bool(identity_testable and expected_label in best_labels)
            if identity_testable:
                direct_order_testable += 1
                direct_order_supported += int(identity_supported)
            else:
                frame_identity_testable = False
            if identity_testable and not identity_supported:
                frame_identity_supported = False
                add_anomaly(
                    anomalies,
                    frame_key,
                    "pose_order_not_best_geometric_match",
                    f"expected={expected_label}:{expected_score:.4f}, best={best_labels}:{best_score:.4f}",
                    index,
                )

            pose_analyses.append(
                {
                    "frame_key": frame_key,
                    "split": split,
                    "surgery_type": surgery_type,
                    "procedure_id": procedure_id,
                    "frame_id": frame_id,
                    "instance_index": index,
                    **keypoint_values,
                    "has_off_image_keypoint": instance_off_image,
                    "skeleton_sample_count": len(samples),
                    "expected_mask_overlap": expected_score,
                    "best_mask_overlap": best_score,
                    "best_mask_labels": json.dumps(best_labels),
                    "identity_testable": identity_testable,
                    "identity_supported": identity_supported,
                    "score_by_mask_label": scores,
                    "edges_json": json.dumps(edges, separators=(",", ":")),
                    "transitions_json": json.dumps(transitions, separators=(",", ":")),
                }
            )

        assignments = optimal_assignment(
            [analysis["score_by_mask_label"] for analysis in pose_analyses], labels
        )
        assigned_labels = {label for label in assignments if label is not None}
        reliable_mapping_count = 0
        for analysis, matched_label in zip(pose_analyses, assignments):
            scores = analysis.pop("score_by_mask_label")
            matched_score = scores.get(matched_label, 0.0) if matched_label is not None else 0.0
            alternatives = [score for label, score in scores.items() if label != matched_label]
            next_best_score = max(alternatives, default=0.0)
            margin = matched_score - next_best_score
            # This threshold is intentionally conservative and is reported,
            # not silently used to discard instances downstream.
            mapping_reliable = matched_label is not None and matched_score >= 0.25 and margin >= 0.05
            reliable_mapping_count += int(mapping_reliable)
            if not mapping_reliable:
                add_anomaly(
                    anomalies,
                    frame_key,
                    "ambiguous_pose_mask_mapping",
                    f"assigned={matched_label}, score={matched_score:.4f}, margin={margin:.4f}",
                    analysis["instance_index"],
                )
            mask_info = stats.get(matched_label, {}) if matched_label is not None else {}
            instrument_rows.append(
                {
                    **analysis,
                    "matched_mask_label": matched_label,
                    "matched_mask_present": matched_label in stats if matched_label is not None else False,
                    **mask_info,
                    "matched_mask_overlap": matched_score,
                    "mapping_margin": margin,
                    "mapping_reliable": mapping_reliable,
                    "all_mask_overlap_scores": json.dumps(scores, sort_keys=True),
                }
            )
        reverse_assignment = {label: index for index, label in enumerate(assignments) if label is not None}
        for label in labels:
            mask_rows.append(
                {
                    "frame_key": frame_key,
                    "split": split,
                    "surgery_type": surgery_type,
                    "procedure_id": procedure_id,
                    "frame_id": frame_id,
                    "mask_label": label,
                    **stats[label],
                    "matched_pose_index": reverse_assignment.get(label),
                    "matched": label in assigned_labels,
                }
            )

        frame_rows.append(
            {
                "frame_key": frame_key,
                "split": split,
                "surgery_type": surgery_type,
                "procedure_id": procedure_id,
                "frame_id": frame_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "pose_path": str(pose_path),
                "width": mask.width,
                "height": mask.height,
                "pose_instance_count": len(poses),
                "mask_instance_count": len(labels),
                "mask_labels": json.dumps(labels),
                "counts_match": counts_match,
                "canonical_mask_label_set": canonical_label_set,
                "has_off_image_keypoint": frame_off_image,
                "has_missing_keypoint": frame_missing,
                "identity_testable_for_all_instances": frame_identity_testable,
                "identity_supported_for_all_testable": frame_identity_supported,
                "geometrically_matched_pose_count": len(assigned_labels),
                "reliably_matched_pose_count": reliable_mapping_count,
            }
        )
        if number % 500 == 0:
            print(f"Processed {number}/{len(pose_paths)} frames", file=sys.stderr, flush=True)

    frame_fields = list(frame_rows[0]) if frame_rows else []
    instrument_fields: list[str] = []
    for row in instrument_rows:
        for field in row:
            if field not in instrument_fields:
                instrument_fields.append(field)
    anomaly_fields = ["frame_key", "instance_index", "code", "detail"]
    write_csv(output_dir / "frames.csv", frame_rows, frame_fields)
    write_csv(output_dir / "instruments.csv", instrument_rows, instrument_fields)
    mask_fields = list(mask_rows[0]) if mask_rows else []
    write_csv(output_dir / "masks.csv", mask_rows, mask_fields)
    write_csv(output_dir / "anomalies.csv", anomalies, anomaly_fields)
    anomaly_counts = collections.Counter(row["code"] for row in anomalies)
    summary = {
        "root": str(root),
        "frame_count": len(frame_rows),
        "instrument_count": len(instrument_rows),
        "mask_instance_count": len(mask_rows),
        "zero_pose_frame_count": sum(row["pose_instance_count"] == 0 for row in frame_rows),
        "pose_mask_count_match_frames": sum(row["counts_match"] for row in frame_rows),
        "canonical_mask_label_set_match_frames": sum(row["canonical_mask_label_set"] for row in frame_rows),
        "off_image_frame_count": sum(row["has_off_image_keypoint"] for row in frame_rows),
        "off_image_instance_count": sum(row["has_off_image_keypoint"] for row in instrument_rows),
        "missing_keypoint_frame_count": sum(row["has_missing_keypoint"] for row in frame_rows),
        "direct_order_testable_instances": direct_order_testable,
        "direct_order_supported_instances": direct_order_supported,
        "direct_order_support_rate": direct_order_supported / direct_order_testable if direct_order_testable else None,
        "geometrically_matched_pose_count": sum(row["matched_mask_label"] not in (None, "") for row in instrument_rows),
        "reliably_matched_pose_count": sum(row["mapping_reliable"] for row in instrument_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "pose_count_distribution": {str(key): value for key, value in sorted(pose_count_distribution.items())},
        "visibility_tag_counts": dict(sorted(tag_counts.items())),
        "missing_by_keypoint": dict(sorted(missing_by_keypoint.items())),
        "off_image_by_keypoint": dict(sorted(off_image_by_keypoint.items())),
        "mask_label_occurrences": {str(key): value for key, value in sorted(mask_label_counts.items())},
        "anomaly_count": len(anomalies),
        "anomaly_counts": dict(sorted(anomaly_counts.items())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "schema_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/RobustMIPS"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests/c2"))
    args = parser.parse_args()
    if not args.root.is_dir():
        print(f"Dataset root not found: {args.root}", file=sys.stderr)
        return 2
    summary = build(args.root, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
