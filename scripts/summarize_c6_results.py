#!/usr/bin/env python3
"""Regenerate C6 summary.json from retained per-instance results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from evaluate_masks import DEFAULT_THRESHOLDS, summarize_ious


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("outputs/c6_sam1_vit_h_pilot/per_instance_results.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.results.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row["iou"])
        groups[f"strategy:{row['strategy']}:overall"].append(value)
        groups[f"strategy:{row['strategy']}:bucket_{row['bucket'].lower()}"] .append(value)
    summary = {name: summarize_ious(values, DEFAULT_THRESHOLDS) for name, values in sorted(groups.items())}
    output = args.output or args.results.parent / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} groups to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
