#!/usr/bin/env python3
"""Render one comparable manuscript panel for all seven C5 prompt styles."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path


STRATEGIES = [
    ("p1_visible_points", "P1 Visible points"),
    ("p2_visible_skeleton", "P2 Visible skeleton"),
    ("p3_visible_box", "P3 Visible-pose box"),
    ("p4_visible_points_box", "P4 Points + box"),
    ("p5_visible_points_negatives", "P5 Points + negatives"),
    ("p6_visible_points_box_negatives", "P6 Points + box + negatives"),
    ("p7_shifted_entry_ablation", "P7 Shifted-entry ablation"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("data/manifests/c5_prompts/prompts.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("outputs/c6_sam1_vit_h_pilot/per_instance_results.csv"))
    parser.add_argument("--pilot-index", type=int, default=109)
    parser.add_argument("--pose-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manuscript_prompt_styles"))
    args = parser.parse_args()
    prefix = f"{args.pilot_index}:{args.pose_index}:"
    prompts = {}
    with args.prompts.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["prompt_id"].startswith(prefix):
                prompts[row["strategy"]] = row
    with args.results.open(newline="", encoding="utf-8") as stream:
        results = {row["strategy"]: row for row in csv.DictReader(stream) if row["prompt_id"].startswith(prefix)}
    if set(prompts) != {strategy for strategy, _ in STRATEGIES}:
        raise RuntimeError("selected instance does not contain all seven strategies")
    first = prompts[STRATEGIES[0][0]]
    raw = Path(first["image_path"]).read_bytes()
    image_href = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    image_width, image_height = 960, 540
    panel_w, image_w = 430, 410
    scale = image_w / image_width
    image_h = image_height * scale
    panel_h = 292
    canvas_w, canvas_h = panel_w * 3, panel_h * 3 + 76
    source_poses = json.loads(Path(first["image_path"]).with_name("toolposes.json").read_text())
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:17px;font-weight:bold}.sub{font-size:13px}.legend{font-size:14px}.panel{fill:white;stroke:#bbb}.box{fill:none;stroke:#ffd600;stroke-width:4}.pos{fill:#7cff00;stroke:#111;stroke-width:1.5}.neg{stroke:#ff2848;stroke-width:4}.shift{fill:#c77dff;stroke:#111;stroke-width:2}.source-edge{stroke-width:4;stroke-linecap:round}.source-visible{stroke:#111;stroke-width:1.5}.source-occluded{fill:white;stroke-width:3;stroke-dasharray:3,2}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<defs><image id="raw" href="{image_href}" width="{image_w}" height="{image_h}"/><clipPath id="source-clip"><rect width="{image_w}" height="{image_h}"/></clipPath></defs>',
    ]
    # Panel 1: complete, unfiltered source annotation before prompt filtering.
    x0 = y0 = 0
    ix, iy = x0 + 10, y0 + 43
    pieces += [f'<rect class="panel" x="4" y="4" width="{panel_w-8}" height="{panel_h-8}"/>',
               '<text class="title" x="10" y="22">Source: complete annotated skeletons</text>',
               '<text class="sub" x="10" y="39">before visibility and bounds filtering</text>',
               f'<g transform="translate({ix},{iy})" clip-path="url(#source-clip)"><use href="#raw"/>']
    source_colors = ("#00e5ff", "#ff3d71", "#7cff00", "#ffd600")
    for pose_index, pose in enumerate(source_poses):
        color = source_colors[pose_index % len(source_colors)]
        nodes, tags = pose.get("nodes", []), pose.get("tags", [])
        for edge in pose.get("edges", []):
            if len(edge) == 2 and max(edge) < len(nodes) and nodes[edge[0]] is not None and nodes[edge[1]] is not None:
                a, b = nodes[edge[0]], nodes[edge[1]]
                pieces.append(f'<line class="source-edge" stroke="{color}" x1="{a[0]*scale:.2f}" y1="{a[1]*scale:.2f}" x2="{b[0]*scale:.2f}" y2="{b[1]*scale:.2f}"/>')
        for node, tag in zip(nodes, tags):
            if node is None:
                continue
            cls = "source-occluded" if tag == "occluded" else "source-visible"
            fill = "white" if tag == "occluded" else color
            pieces.append(f'<circle class="{cls}" fill="{fill}" stroke="{color}" cx="{node[0]*scale:.2f}" cy="{node[1]*scale:.2f}" r="6"/>')
    pieces += ['</g>', '<text class="sub" x="16" y="280">All edges retained; missing nodes have no coordinate.</text>']
    for index, (strategy, title) in enumerate(STRATEGIES):
        cell = index + 1
        col, row = cell % 3, cell // 3
        x0, y0 = col * panel_w, row * panel_h
        ix, iy = x0 + 10, y0 + 43
        prompt, result = prompts[strategy], results[strategy]
        pieces += [f'<rect class="panel" x="{x0+4}" y="{y0+4}" width="{panel_w-8}" height="{panel_h-8}"/>',
                   f'<text class="title" x="{x0+10}" y="{y0+22}">{html.escape(title)}</text>',
                   f'<text class="sub" x="{x0+10}" y="{y0+39}">deployable mask IoU = {float(result["iou"]):.3f}</text>',
                   f'<g transform="translate({ix},{iy})"><use href="#raw"/>']
        if prompt["box_xyxy"]:
            x1, y1, x2, y2 = prompt["box_xyxy"]
            pieces.append(f'<rect class="box" x="{x1*scale:.2f}" y="{y1*scale:.2f}" width="{(x2-x1)*scale:.2f}" height="{(y2-y1)*scale:.2f}"/>')
        shifted = prompt["shifted_entry_point"]
        for x, y in prompt["positive_points"]:
            if shifted and abs(x-shifted[0]) < 1e-6 and abs(y-shifted[1]) < 1e-6:
                pieces.append(f'<rect class="shift" x="{x*scale-6:.2f}" y="{y*scale-6:.2f}" width="12" height="12" transform="rotate(45 {x*scale:.2f} {y*scale:.2f})"/>')
            else:
                pieces.append(f'<circle class="pos" cx="{x*scale:.2f}" cy="{y*scale:.2f}" r="5"/>')
        for x, y in prompt["negative_points"]:
            sx, sy = x * scale, y * scale
            pieces += [f'<line class="neg" x1="{sx-6:.2f}" y1="{sy-6:.2f}" x2="{sx+6:.2f}" y2="{sy+6:.2f}"/>',
                       f'<line class="neg" x1="{sx-6:.2f}" y1="{sy+6:.2f}" x2="{sx+6:.2f}" y2="{sy-6:.2f}"/>']
        pieces.append('</g>')
    # The ninth cell is a large legend/caption block.
    x0, y0 = 2 * panel_w, 2 * panel_h
    pieces += [f'<rect class="panel" x="{x0+4}" y="{y0+4}" width="{panel_w-8}" height="{panel_h-8}"/>',
               f'<text class="title" x="{x0+18}" y="{y0+28}">Prompt and source legend</text>',
               f'<circle class="pos" cx="{x0+26}" cy="{y0+66}" r="6"/><text class="legend" x="{x0+44}" y="{y0+71}">positive point</text>',
               f'<line class="neg" x1="{x0+20}" y1="{y0+91}" x2="{x0+32}" y2="{y0+103}"/><line class="neg" x1="{x0+20}" y1="{y0+103}" x2="{x0+32}" y2="{y0+91}"/><text class="legend" x="{x0+44}" y="{y0+102}">other-instrument negative</text>',
               f'<rect class="box" x="{x0+19}" y="{y0+121}" width="16" height="16"/><text class="legend" x="{x0+44}" y="{y0+135}">visible-pose box</text>',
               f'<rect class="shift" x="{x0+20}" y="{y0+157}" width="12" height="12" transform="rotate(45 {x0+26} {y0+163})"/><text class="legend" x="{x0+44}" y="{y0+168}">derived shifted EntryPoint</text>',
               f'<circle cx="{x0+26}" cy="{y0+195}" r="6" fill="white" stroke="#00e5ff" stroke-width="3" stroke-dasharray="3,2"/><text class="legend" x="{x0+44}" y="{y0+200}">occluded source keypoint</text>',
               f'<text class="sub" x="{x0+18}" y="{y0+225}">Frame: {html.escape(first["frame_key"])}</text>',
               f'<text class="sub" x="{x0+18}" y="{y0+245}">Target pose {first["pose_index"]}; Bucket {first["bucket"]}</text>',
               f'<text class="sub" x="{x0+18}" y="{y0+267}">Same image and target in every prompt panel.</text>',
               f'<text class="legend" x="20" y="{canvas_h-28}">Figure: Complete source pose annotation followed by seven pose-to-SAM prompt encodings for one multi-instrument, off-image case.</text>',
               '</svg>']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "sam_prompt_strategies.svg"
    destination.write_text("\n".join(pieces) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Manuscript prompt-strategy example\n\n"
        "The first panel shows the complete unfiltered source skeleton annotations. All seven prompt panels use the same target pose from `Training/Rectal resection/1/34500`. "
        "Green circles are positive prompts, red crosses are competing-instrument negatives, "
        "yellow is the visible-pose box, and the purple diamond is the derived shifted EntryPoint.\n\n"
        "![Prompt strategies](sam_prompt_strategies.svg)\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
