#!/usr/bin/env python3
"""Generate dependency-free SVG audit overlays from the C2 manifests."""

from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import random
import urllib.parse
from pathlib import Path
from typing import Any


COLORS = ("#00e5ff", "#ff3d71", "#7cff00", "#ffd600", "#c77dff", "#ff8c42", "#00f5a0")
KEYPOINT_NAMES = ("entry", "hinge", "tip1", "tip2")


def truth(value: str) -> bool:
    return value.lower() == "true"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def choose_frames(
    frames: list[dict[str, str]], anomalies: list[dict[str, str]], per_category: int, seed: int
) -> dict[str, list[str]]:
    by_code: dict[str, list[str]] = collections.defaultdict(list)
    for row in anomalies:
        if row["frame_key"] not in by_code[row["code"]]:
            by_code[row["code"]].append(row["frame_key"])
    rng = random.Random(seed)
    ordinary = [
        row["frame_key"]
        for row in frames
        if truth(row["counts_match"])
        and truth(row["identity_supported_for_all_testable"])
        and not truth(row["has_off_image_keypoint"])
        and int(row["pose_instance_count"]) > 0
    ]
    rng.shuffle(ordinary)
    off_image = [row["frame_key"] for row in frames if truth(row["has_off_image_keypoint"])]
    rng.shuffle(off_image)
    categories = {
        "ordinary": ordinary,
        "off_image": off_image,
        "swapped_order": by_code["pose_order_not_best_geometric_match"],
        "count_mismatch": by_code["pose_mask_count_mismatch"],
        "ambiguous_mapping": by_code["ambiguous_pose_mask_mapping"],
    }
    return {name: values[:per_category] for name, values in categories.items()}


def coordinate(row: dict[str, str], name: str) -> tuple[float, float] | None:
    x, y = row.get(f"{name}_x", ""), row.get(f"{name}_y", "")
    if x == "" or y == "":
        return None
    return float(x), float(y)


def svg_for_frame(
    frame: dict[str, str], instruments: list[dict[str, str]], output_path: Path, repo_root: Path
) -> None:
    width, height = int(frame["width"]), int(frame["height"])
    gap = 20
    panel_width = width
    canvas_width = panel_width * 2 + gap
    header = 74
    canvas_height = height + header

    def href(path_text: str) -> str:
        relative = Path(path_text).resolve().relative_to(repo_root.resolve())
        # SVG URL references need URI escaping for surgery names containing spaces.
        return urllib.parse.quote(str(Path("../..") / relative), safe="/._-")

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" '
        f'viewBox="0 0 {canvas_width} {canvas_height}">',
        "<style>text{font-family:monospace;fill:white}.label{font-size:17px;font-weight:bold}"
        ".small{font-size:13px}.edge{stroke-width:4;stroke-linecap:round}.node{stroke:black;stroke-width:2}</style>",
        '<rect width="100%" height="100%" fill="#171717"/>',
        f'<text class="label" x="10" y="22">{html.escape(frame["frame_key"])}</text>',
        f'<text class="small" x="10" y="44">poses={frame["pose_instance_count"]}, '
        f'masks={frame["mask_instance_count"]}, labels={html.escape(frame["mask_labels"])}</text>',
        f'<text class="small" x="10" y="64">left: raw image; right: reference instance mask</text>',
        f'<image x="0" y="{header}" width="{width}" height="{height}" href="{href(frame["image_path"])}"/>',
        f'<rect x="{width + gap}" y="{header}" width="{width}" height="{height}" fill="black"/>',
        f'<image x="{width + gap}" y="{header}" width="{width}" height="{height}" '
        f'href="{href(frame["mask_path"])}"/>',
    ]
    for pose_index, row in enumerate(sorted(instruments, key=lambda item: int(item["instance_index"]))):
        color = COLORS[pose_index % len(COLORS)]
        nodes = [coordinate(row, name) for name in KEYPOINT_NAMES]
        edges = json.loads(row["edges_json"])
        for panel_x in (0, width + gap):
            pieces.append(f'<g clip-path="url(#clip{panel_x})">')
            for a, b in edges:
                if a >= len(nodes) or b >= len(nodes) or nodes[a] is None or nodes[b] is None:
                    continue
                ax, ay = nodes[a]
                bx, by = nodes[b]
                pieces.append(
                    f'<line class="edge" x1="{panel_x + ax:.2f}" y1="{header + ay:.2f}" '
                    f'x2="{panel_x + bx:.2f}" y2="{header + by:.2f}" stroke="{color}"/>'
                )
            for node_index, node in enumerate(nodes):
                if node is None:
                    continue
                x, y = node
                visibility = row[f"{KEYPOINT_NAMES[node_index]}_visibility"]
                radius = 7 if visibility == "visible" else 5
                dash = ' stroke-dasharray="3,2"' if visibility != "visible" else ""
                pieces.append(
                    f'<circle class="node" cx="{panel_x + x:.2f}" cy="{header + y:.2f}" r="{radius}" '
                    f'fill="{color}"{dash}/>'
                )
            pieces.append("</g>")
        label = row.get("matched_mask_label", "") or "unmatched"
        score = float(row.get("matched_mask_overlap", 0) or 0)
        reliable = row.get("mapping_reliable", "False")
        pieces.append(
            f'<text class="small" x="{width + gap + 8}" y="{header + 18 + pose_index * 17}" '
            f'fill="{color}" style="fill:{color}">pose {row["instance_index"]} → mask {label}; '
            f'overlap={score:.3f}; reliable={reliable}</text>'
        )
    pieces.insert(
        3,
        f'<defs><clipPath id="clip0"><rect x="0" y="{header}" width="{width}" height="{height}"/></clipPath>'
        f'<clipPath id="clip{width + gap}"><rect x="{width + gap}" y="{header}" width="{width}" '
        f'height="{height}"/></clipPath></defs>',
    )
    pieces.append("</svg>")
    output_path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests/c2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/c2_overlays"))
    parser.add_argument("--per-category", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frames = load_rows(args.manifest_dir / "frames.csv")
    instruments = load_rows(args.manifest_dir / "instruments.csv")
    anomalies = load_rows(args.manifest_dir / "anomalies.csv")
    frame_lookup = {row["frame_key"]: row for row in frames}
    instrument_lookup: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in instruments:
        instrument_lookup[row["frame_key"]].append(row)
    selected = choose_frames(frames, anomalies, args.per_category, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path.cwd()
    index_parts = [
        "<!doctype html><meta charset='utf-8'><title>ROBUST-MIPS C2 audit</title>",
        "<style>body{background:#111;color:#eee;font-family:sans-serif}object{width:100%;max-width:1940px;"
        "height:640px;border:1px solid #555;margin-bottom:24px}h2{margin-top:40px}</style>",
        "<h1>ROBUST-MIPS C2 pose-to-mask audit</h1>",
    ]
    selection_manifest: list[dict[str, str]] = []
    for category, frame_keys in selected.items():
        index_parts.append(f"<h2>{html.escape(category)}</h2>")
        for order, frame_key in enumerate(frame_keys):
            filename = f"{category}_{order:02d}.svg"
            svg_for_frame(
                frame_lookup[frame_key], instrument_lookup.get(frame_key, []), args.output_dir / filename, repo_root
            )
            index_parts.append(f"<h3>{html.escape(frame_key)}</h3><object data='{filename}'></object>")
            selection_manifest.append({"category": category, "frame_key": frame_key, "overlay": filename})
    (args.output_dir / "index.html").write_text("\n".join(index_parts) + "\n", encoding="utf-8")
    with (args.output_dir / "selection.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["category", "frame_key", "overlay"])
        writer.writeheader()
        writer.writerows(selection_manifest)
    print(f"Generated {len(selection_manifest)} overlays in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
