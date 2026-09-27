#!/usr/bin/env python3
"""Create dependency-free SVG violin/box plots for C6 per-instance IoUs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ORDER = [
    "p1_visible_points", "p2_visible_skeleton", "p3_visible_box",
    "p4_visible_points_box", "p5_visible_points_negatives",
    "p6_visible_points_box_negatives", "p7_shifted_entry_ablation",
]
SHORT = {strategy: strategy.split("_", 1)[0].upper() for strategy in ORDER}
COLORS = ["#00bcd4", "#ff3d71", "#7cb342", "#ffc107", "#8e67cc", "#ff8c42", "#00a878"]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def density(values: list[float], ys: list[float], bandwidth: float = 0.055) -> list[float]:
    scale = 1 / (bandwidth * math.sqrt(2 * math.pi) * len(values))
    return [scale * sum(math.exp(-0.5 * ((y - value) / bandwidth) ** 2) for value in values) for y in ys]


def panel(groups: dict[str, list[float]], title: str, x0: int, y0: int, width: int, height: int) -> list[str]:
    left, right, top, bottom = 66, 24, 42, 128
    plot_w, plot_h = width - left - right, height - top - bottom
    active = [strategy for strategy in ORDER if groups.get(strategy)]
    sx = plot_w / len(active)
    py = lambda value: y0 + top + (1 - value) * plot_h
    pieces = [f'<text x="{x0 + width/2}" y="{y0 + 22}" text-anchor="middle" class="title">{title}</text>',
              f'<line x1="{x0+left}" y1="{py(0)}" x2="{x0+left+plot_w}" y2="{py(0)}" class="axis"/>',
              f'<line x1="{x0+left}" y1="{py(0)}" x2="{x0+left}" y2="{py(1)}" class="axis"/>']
    for tick in (0, .25, .5, .7, .8, .9, 1):
        y = py(tick)
        pieces += [f'<line x1="{x0+left}" y1="{y}" x2="{x0+left+plot_w}" y2="{y}" class="grid"/>',
                   f'<text x="{x0+left-8}" y="{y+4}" text-anchor="end" class="tick">{tick:.2g}</text>']
    ys = [i / 100 for i in range(101)]
    for index, strategy in enumerate(active):
        values = groups.get(strategy, [])
        if not values:
            continue
        cx = x0 + left + (index + .5) * sx
        ds = density(values, ys)
        max_d = max(ds) or 1
        half = sx * .34
        right_side = [(cx + half * d / max_d, py(y)) for y, d in zip(ys, ds)]
        left_side = [(cx - half * d / max_d, py(y)) for y, d in reversed(list(zip(ys, ds)))]
        path = " ".join([f"M {right_side[0][0]:.1f},{right_side[0][1]:.1f}"] +
                        [f"L {x:.1f},{y:.1f}" for x, y in right_side[1:] + left_side] + ["Z"])
        pieces.append(f'<path d="{path}" fill="{COLORS[index]}" fill-opacity="0.48" stroke="{COLORS[index]}"/>')
        q1, median, q3 = quantile(values, .25), quantile(values, .5), quantile(values, .75)
        low, high = min(values), max(values)
        box_half = sx * .12
        pieces += [f'<line x1="{cx}" y1="{py(low)}" x2="{cx}" y2="{py(high)}" class="whisker"/>',
                   f'<rect x="{cx-box_half}" y="{py(q3)}" width="{2*box_half}" height="{py(q1)-py(q3)}" class="box"/>',
                   f'<line x1="{cx-box_half}" y1="{py(median)}" x2="{cx+box_half}" y2="{py(median)}" class="median"/>',
                   f'<circle cx="{cx}" cy="{py(statistics.fmean(values))}" r="3.5" class="mean"/>',
                   f'<text x="{cx}" y="{y0+height-78}" text-anchor="middle" class="label">{SHORT[strategy]}</text>',
                   f'<text x="{cx}" y="{y0+height-57}" text-anchor="middle" class="n">n={len(values)}</text>']
    return pieces


def write_plot(path: Path, panels: list[tuple[dict[str, list[float]], str]], panel_width: int = 650) -> None:
    height, width = 620, panel_width * len(panels)
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<style>text{font-family:sans-serif;fill:#222}.title{font-size:18px;font-weight:bold}.tick,.n{font-size:11px}.label{font-size:13px;font-weight:bold}.axis{stroke:#222;stroke-width:1.5}.grid{stroke:#ddd}.whisker{stroke:#222}.box{fill:white;fill-opacity:.75;stroke:#222}.median{stroke:#111;stroke-width:3}.mean{fill:#111}</style>',
              '<rect width="100%" height="100%" fill="white"/>']
    for index, (groups, title) in enumerate(panels):
        pieces.extend(panel(groups, title, index * panel_width, 0, panel_width, height))
    pieces += [f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">Per-instance mask IoU</text>',
               f'<text x="32" y="{height-14}" class="n">Violin = distribution; box = IQR/median; dot = mean; whisker = range</text>', "</svg>"]
    path.write_text("\n".join(pieces) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("outputs/c6_sam1_vit_h_pilot/per_instance_results.csv"))
    args = parser.parse_args()
    with args.results.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    overall: dict[str, list[float]] = defaultdict(list)
    buckets: dict[str, dict[str, list[float]]] = {bucket: defaultdict(list) for bucket in "ABC"}
    for row in rows:
        value = float(row["iou"])
        overall[row["strategy"]].append(value)
        buckets[row["bucket"]][row["strategy"]].append(value)
    destination = args.results.parent / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    write_plot(destination / "iou_violin_box_overall.svg", [(overall, "All pilot instances")], panel_width=900)
    write_plot(destination / "iou_violin_box_by_bucket.svg",
               [(buckets[bucket], f"Bucket {bucket}") for bucket in "ABC"], panel_width=720)
    print(f"Wrote plots to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
