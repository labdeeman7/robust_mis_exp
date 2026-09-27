#!/usr/bin/env python3
"""Audit the ROBUST-MIPS ZIP structure without extracting its image payload."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import struct
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


EXPECTED_FRAME_FILES = {
    "raw.png": "image",
    "instrument_instances.png": "mask",
    # The paper's schematic calls this raw.json; release v1 uses toolposes.json.
    "toolposes.json": "pose",
}


@dataclass(frozen=True)
class FrameRecord:
    frame_dir: str
    split: str | None
    surgery_type: str | None
    procedure_id: str | None
    frame_id: str | None
    image_member: str | None
    mask_member: str | None
    pose_member: str | None
    image_width: int | None
    image_height: int | None
    mask_width: int | None
    mask_height: int | None
    pose_top_level_type: str | None
    pose_instance_count: int | None
    complete_triplet: bool


def png_dimensions(source: zipfile.ZipFile, member: str) -> tuple[int, int]:
    with source.open(member) as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a valid PNG header: {member}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def infer_hierarchy(frame_dir: PurePosixPath) -> tuple[str | None, str | None, str | None, str | None]:
    parts = frame_dir.parts
    # The paper describes Split/Surgery_type/Procedure_ID/Frame_ID. Archives
    # sometimes add one leading wrapper directory, so parse from the tail.
    if len(parts) < 4:
        padded: list[str | None] = [None] * (4 - len(parts)) + list(parts)
        return tuple(padded)  # type: ignore[return-value]
    return parts[-4], parts[-3], parts[-2], parts[-1]


def infer_pose_count(payload: Any) -> int | None:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return None
    for key in ("instances", "instruments", "annotations", "objects", "shapes"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    # Some annotation tools store one instrument directly in each JSON file.
    if "nodes" in payload or "keypoints" in payload:
        return 1
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, records: Iterable[FrameRecord]) -> None:
    rows = [asdict(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(FrameRecord.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)


def audit(archive: Path, output_dir: Path, expected_frames: int) -> dict[str, Any]:
    members_by_dir: dict[PurePosixPath, dict[str, str]] = collections.defaultdict(dict)
    basename_counts: collections.Counter[str] = collections.Counter()
    extension_counts: collections.Counter[str] = collections.Counter()
    duplicate_members: list[str] = []
    corrupt_members: list[str] = []
    seen: set[str] = set()

    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if info.is_dir():
                continue
            member = PurePosixPath(info.filename)
            # Release v1 contains macOS AppleDouble/resource-fork entries for
            # nearly every payload. They are packaging metadata, not samples.
            if "__MACOSX" in member.parts or member.name.startswith("._"):
                continue
            if info.filename in seen:
                duplicate_members.append(info.filename)
            seen.add(info.filename)
            basename_counts[member.name] += 1
            extension_counts[member.suffix.lower()] += 1
            if member.name in EXPECTED_FRAME_FILES:
                members_by_dir[member.parent][EXPECTED_FRAME_FILES[member.name]] = info.filename

        # CRC check streams every member but does not extract it.
        bad_member = source.testzip()
        if bad_member:
            corrupt_members.append(bad_member)

        records: list[FrameRecord] = []
        pose_schema_examples: dict[str, Any] = {}
        json_errors: list[dict[str, str]] = []
        dimension_errors: list[dict[str, str]] = []
        for frame_dir, roles in sorted(members_by_dir.items(), key=lambda item: str(item[0])):
            split, surgery_type, procedure_id, frame_id = infer_hierarchy(frame_dir)
            image_size: tuple[int | None, int | None] = (None, None)
            mask_size: tuple[int | None, int | None] = (None, None)
            pose_type = None
            pose_count = None
            for role, member in roles.items():
                try:
                    if role == "image":
                        image_size = png_dimensions(source, member)
                    elif role == "mask":
                        mask_size = png_dimensions(source, member)
                    elif role == "pose":
                        with source.open(member) as stream:
                            payload = json.load(stream)
                        pose_type = type(payload).__name__
                        pose_count = infer_pose_count(payload)
                        if len(pose_schema_examples) < 5:
                            if isinstance(payload, dict):
                                pose_schema_examples[member] = {
                                    "top_level_type": pose_type,
                                    "keys": sorted(payload),
                                    "inferred_instance_count": pose_count,
                                }
                            else:
                                pose_schema_examples[member] = {
                                    "top_level_type": pose_type,
                                    "inferred_instance_count": pose_count,
                                }
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    target = json_errors if role == "pose" else dimension_errors
                    target.append({"member": member, "error": str(error)})

            records.append(
                FrameRecord(
                    frame_dir=str(frame_dir),
                    split=split,
                    surgery_type=surgery_type,
                    procedure_id=procedure_id,
                    frame_id=frame_id,
                    image_member=roles.get("image"),
                    mask_member=roles.get("mask"),
                    pose_member=roles.get("pose"),
                    image_width=image_size[0],
                    image_height=image_size[1],
                    mask_width=mask_size[0],
                    mask_height=mask_size[1],
                    pose_top_level_type=pose_type,
                    pose_instance_count=pose_count,
                    complete_triplet=set(roles) == {"image", "mask", "pose"},
                )
            )

    complete = sum(record.complete_triplet for record in records)
    mismatched_dimensions = sum(
        record.image_width is not None
        and record.mask_width is not None
        and (record.image_width, record.image_height) != (record.mask_width, record.mask_height)
        for record in records
    )
    split_counts = collections.Counter(record.split for record in records)
    report = {
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "expected_frame_count": expected_frames,
        "detected_frame_directories": len(records),
        "complete_triplets": complete,
        "frame_count_matches_expectation": len(records) == expected_frames,
        "all_frames_are_complete_triplets": complete == len(records),
        "mismatched_image_mask_dimensions": mismatched_dimensions,
        "basename_counts": dict(sorted(basename_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "split_counts": {str(key): value for key, value in sorted(split_counts.items(), key=lambda x: str(x[0]))},
        "duplicate_member_count": len(duplicate_members),
        "duplicate_members": duplicate_members,
        "corrupt_members": corrupt_members,
        "json_errors": json_errors,
        "dimension_errors": dimension_errors,
        "pose_schema_examples": pose_schema_examples,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "frames.csv", records)
    (output_dir / "archive_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", nargs="?", type=Path, default=Path("data/archives/RobustMIPS.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests/c1_audit"))
    parser.add_argument("--expected-frames", type=int, default=10_040)
    args = parser.parse_args()
    if not args.archive.is_file():
        print(f"Archive not found: {args.archive}", file=sys.stderr)
        return 2
    try:
        report = audit(args.archive, args.output_dir, args.expected_frames)
    except zipfile.BadZipFile as error:
        print(f"Invalid or incomplete ZIP: {error}", file=sys.stderr)
        return 3
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["frame_count_matches_expectation"] and report["all_frames_are_complete_triplets"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
