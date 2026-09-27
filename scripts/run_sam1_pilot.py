#!/usr/bin/env python3
"""Run frozen SAM 1 over the C4 pilot and evaluate every C5 prompt."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_masks import DEFAULT_THRESHOLDS, summarize_ious


RESULT_FIELDS = (
    "prompt_id", "pilot_index", "bucket", "frame_key", "pose_index", "mask_label", "strategy",
    "positive_point_count", "negative_point_count", "has_box", "has_shifted_entry",
    "no_valid_prompt",
    "selected_candidate", "predicted_quality", "iou", "oracle_candidate", "oracle_iou",
    "candidate_predicted_qualities", "candidate_ious", "inference_seconds",
    "prediction_path",
)


def iou_numpy(prediction: np.ndarray, reference: np.ndarray) -> float:
    intersection = np.logical_and(prediction, reference).sum(dtype=np.int64)
    union = np.logical_or(prediction, reference).sum(dtype=np.int64)
    if reference.sum(dtype=np.int64) == 0:
        raise ValueError("reference instance is empty")
    return float(intersection / union)


def load_prompts(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            grouped[row["frame_key"]].append(row)
    return grouped


def aggregate(rows: list[dict[str, str]]) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row["iou"])
        groups[f"strategy:{row['strategy']}:overall"].append(value)
        groups[f"strategy:{row['strategy']}:bucket_{row['bucket'].lower()}"] .append(value)
    return {name: summarize_ious(values, DEFAULT_THRESHOLDS) for name, values in sorted(groups.items())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("data/manifests/c5_prompts/prompts.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", default="vit_h", choices=("vit_b", "vit_l", "vit_h"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/c6_sam1_vit_h_pilot"))
    parser.add_argument("--segment-anything-path", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.segment_anything_path:
        sys.path.insert(0, str(args.segment_anything_path.resolve()))
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_prompts(args.prompts)
    model = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint)).to(args.device)
    model.eval()
    predictor = SamPredictor(model)
    result_path = args.output_dir / "per_instance_results.csv"
    rows: list[dict[str, str]] = []
    with result_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for frame_number, (frame_key, prompts) in enumerate(sorted(grouped.items()), 1):
            image = np.asarray(Image.open(prompts[0]["image_path"]).convert("RGB"))
            mask_labels = np.asarray(Image.open(prompts[0]["mask_path"]).convert("L"))
            predictor.set_image(image)
            for prompt in prompts:
                positive = prompt["positive_points"]
                negative = prompt["negative_points"]
                points = positive + negative
                point_coords = np.asarray(points, dtype=np.float32) if points else None
                point_labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32) if points else None
                box = np.asarray(prompt["box_xyxy"], dtype=np.float32) if prompt["box_xyxy"] else None
                reference = mask_labels == int(prompt["mask_label"])
                started = time.perf_counter()
                no_valid_prompt = point_coords is None and box is None
                if no_valid_prompt:
                    masks = np.zeros((3, *reference.shape), dtype=bool)
                    qualities = np.zeros(3, dtype=np.float32)
                else:
                    with torch.inference_mode():
                        masks, qualities, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                                                box=box, multimask_output=True)
                elapsed = time.perf_counter() - started
                candidate_ious = [iou_numpy(candidate, reference) for candidate in masks]
                selected = int(np.argmax(qualities))
                oracle = int(np.argmax(candidate_ious))
                prediction_path = prediction_dir / f"{prompt['pilot_index']:03d}_{prompt['pose_index']}_{prompt['strategy']}.png"
                Image.fromarray(masks[selected].astype(np.uint8) * 255, mode="L").save(prediction_path)
                row = {
                    "prompt_id": prompt["prompt_id"], "pilot_index": prompt["pilot_index"],
                    "bucket": prompt["bucket"], "frame_key": frame_key, "pose_index": prompt["pose_index"],
                    "mask_label": prompt["mask_label"], "strategy": prompt["strategy"],
                    "positive_point_count": len(positive), "negative_point_count": len(negative),
                    "has_box": box is not None, "has_shifted_entry": prompt["shifted_entry_point"] is not None,
                    "no_valid_prompt": no_valid_prompt,
                    "selected_candidate": selected, "predicted_quality": f"{float(qualities[selected]):.8f}",
                    "iou": f"{candidate_ious[selected]:.8f}", "oracle_candidate": oracle,
                    "oracle_iou": f"{candidate_ious[oracle]:.8f}",
                    "candidate_predicted_qualities": json.dumps([float(value) for value in qualities]),
                    "candidate_ious": json.dumps(candidate_ious), "inference_seconds": f"{elapsed:.6f}",
                    "prediction_path": str(prediction_path),
                }
                writer.writerow(row)
                stream.flush()
                rows.append({key: str(value) for key, value in row.items()})
            print(f"[{frame_number}/{len(grouped)}] {frame_key}", flush=True)
    metadata = {
        "model": "segment_anything", "model_type": args.model_type,
        "checkpoint": str(args.checkpoint), "device": args.device,
        "frame_count": len(grouped), "result_count": len(rows),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(aggregate(rows), indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
