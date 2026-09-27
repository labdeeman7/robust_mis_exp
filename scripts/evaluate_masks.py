#!/usr/bin/env python3
"""C3 instance-mask IoU and threshold summaries."""

from __future__ import annotations

import statistics
from collections.abc import Sequence


DEFAULT_THRESHOLDS = (0.50, 0.70, 0.80, 0.90)
FAILURE_THRESHOLD = 0.20


def binary_mask_iou(prediction: Sequence[Sequence[object]], reference: Sequence[Sequence[object]]) -> float:
    """Return binary IoU; reject shape mismatch and empty reference instances."""
    if len(prediction) != len(reference):
        raise ValueError("prediction and reference heights differ")
    intersection = union = reference_area = 0
    for predicted_row, reference_row in zip(prediction, reference):
        if len(predicted_row) != len(reference_row):
            raise ValueError("prediction and reference widths differ")
        for predicted_value, reference_value in zip(predicted_row, reference_row):
            predicted = bool(predicted_value)
            target = bool(reference_value)
            intersection += predicted and target
            union += predicted or target
            reference_area += target
    if reference_area == 0:
        raise ValueError("reference instance mask is empty")
    return intersection / union


def summarize_ious(ious: Sequence[float], thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> dict:
    """Summarize retained per-instance IoUs using inclusive thresholds."""
    if not ious:
        raise ValueError("cannot summarize an empty IoU collection")
    values = [float(value) for value in ious]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("IoUs must lie in [0, 1]")
    return {
        "count": len(values),
        "mean_iou": statistics.fmean(values),
        "median_iou": statistics.median(values),
        "failure_threshold": FAILURE_THRESHOLD,
        "failure_count_below_0.20": sum(value < FAILURE_THRESHOLD for value in values),
        "failure_rate_below_0.20": sum(value < FAILURE_THRESHOLD for value in values) / len(values),
        "success_rates": {
            f"iou_at_least_{threshold:.2f}": sum(value >= threshold for value in values) / len(values)
            for threshold in thresholds
        },
    }
