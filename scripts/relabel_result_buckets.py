#!/usr/bin/env python3
"""Relabel retained SAM results from an updated instance manifest without rerunning SAM."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    args = parser.parse_args()

    with args.instances.open(newline="", encoding="utf-8") as stream:
        lookup = {(row["frame_key"], row["pose_index"]): row["bucket"] for row in csv.DictReader(stream)}
    with args.results.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise RuntimeError(f"missing CSV header: {args.results}")
    for row in rows:
        key = (row["frame_key"], row["pose_index"])
        if key not in lookup:
            raise KeyError(f"result has no matching manifest instance: {key}")
        row["bucket"] = lookup[key]
    with args.results.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Relabelled {len(rows)} rows in {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
