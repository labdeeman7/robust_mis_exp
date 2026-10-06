#!/usr/bin/env python3
"""Render best/worst SAM 1 pilot examples after C6 inference."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {"reference": (0, 229, 255), "prediction": (255, 61, 113),
          "positive": (124, 255, 0), "negative": (255, 40, 40), "box": (255, 214, 0)}


def blend_mask(image: Image.Image, mask: Image.Image, color: tuple[int, int, int], alpha: float = 0.55) -> Image.Image:
    overlay = Image.new("RGB", image.size, color)
    return Image.composite(Image.blend(image, overlay, alpha), image, mask.convert("1"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("outputs/c6_sam1_vit_h_pilot/per_instance_results.csv"))
    parser.add_argument("--prompts", type=Path, default=Path("data/manifests/c5_prompts/prompts.jsonl"))
    parser.add_argument("--per-tail", type=int, default=3)
    parser.add_argument("--by-bucket", action="store_true",
                        help="Render separate best/worst examples for every strategy and bucket")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.results.parent / ("gallery_by_bucket" if args.by_bucket else "gallery")
    output.mkdir(parents=True, exist_ok=True)
    prompt_by_id = {}
    with args.prompts.open(encoding="utf-8") as stream:
        for line in stream:
            prompt = json.loads(line)
            prompt_by_id[prompt["prompt_id"]] = prompt
    with args.results.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        key = (row["strategy"], row["bucket"] if args.by_bucket else None)
        groups[key].append(row)
    title = "SAM 1 ViT-H gallery by reporting bucket" if args.by_bucket else "SAM 1 ViT-H gallery"
    readme = [f"# {title}", "",
              "Panel order: **raw + prompts | reference (cyan) | prediction (pink)**.",
              "Positive prompts are green, negatives red, and boxes yellow.", ""]
    for (strategy, bucket), strategy_rows in sorted(groups.items()):
        ordered = sorted(strategy_rows, key=lambda row: float(row["iou"]))
        selected = [("worst", row) for row in ordered[:args.per_tail]] + [("best", row) for row in ordered[-args.per_tail:]]
        heading = f"{strategy} — Bucket {bucket}" if bucket else strategy
        readme += [f"## {heading}", ""]
        tail_ranks = defaultdict(int)
        for tail, row in selected:
            tail_ranks[tail] += 1
            rank = tail_ranks[tail]
            prompt = prompt_by_id[row["prompt_id"]]
            raw = Image.open(prompt["image_path"]).convert("RGB")
            labels = Image.open(prompt["mask_path"]).convert("L")
            reference = labels.point(lambda value, target=int(prompt["mask_label"]): 255 if value == target else 0)
            prediction = Image.open(row["prediction_path"]).convert("L")
            prompt_panel = raw.copy()
            draw = ImageDraw.Draw(prompt_panel)
            for x, y in prompt["positive_points"]:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=COLORS["positive"], outline=(0, 0, 0), width=2)
            for x, y in prompt["negative_points"]:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=COLORS["negative"], outline=(0, 0, 0), width=2)
            if prompt["box_xyxy"]:
                draw.rectangle(prompt["box_xyxy"], outline=COLORS["box"], width=4)
            panels = [prompt_panel, blend_mask(raw, reference, COLORS["reference"]),
                      blend_mask(raw, prediction, COLORS["prediction"])]
            canvas = Image.new("RGB", (raw.width * 3 + 12, raw.height), (255, 255, 255))
            for index, panel in enumerate(panels):
                canvas.paste(panel, (index * (raw.width + 6), 0))
            bucket_part = f"bucket_{bucket.lower()}_" if bucket else ""
            filename = f"{strategy}_{bucket_part}{tail}_{rank:02d}_iou_{float(row['iou']):.3f}.png"
            canvas.save(output / filename)
            outcome = "failure" if float(row["iou"]) < 0.20 else ("success" if float(row["iou"]) >= 0.70 else "intermediate")
            readme += [f"### {tail} {rank}: IoU {float(row['iou']):.3f} ({outcome})", "",
                       f"`{row['frame_key']}`, pose {row['pose_index']}, bucket {row['bucket']}", "",
                       f"![{filename}]({filename})", ""]
    (output / "README.md").write_text("\n".join(readme) + "\n")
    print(f"Rendered {sum(1 for _ in output.glob('*.png'))} gallery panels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
