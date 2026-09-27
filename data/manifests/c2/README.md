# C2 normalized annotation manifests

Generated from the immutable ROBUST-MIPS release by:

```bash
python3 scripts/build_c2_manifests.py
python3 scripts/generate_c2_overlays.py
```

## Files

- `frames.csv`: one row per released frame, including split hierarchy, source paths, instance counts, and frame-level quality flags.
- `instruments.csv`: one row per pose instance, including all four keypoints, visibility/off-image flags, skeleton metadata, pose-to-mask overlap scores, and the geometrically matched mask.
- `masks.csv`: one row per nonzero reference-mask label, including area, bounding box, centroid, and matched pose index where available.
- `anomalies.csv`: explicit count, label, ordering, and ambiguous-mapping findings.
- `schema_summary.json`: aggregate counts for C2.

## Critical correspondence rule

The order of objects in `toolposes.json` is **not** a reliable reference-mask
identifier. Among 11,197 pose instances with directly testable visible geometry,
only 8,170 (72.97%) had their best-overlap mask at the naively corresponding
label (`50`, `100`, ...).

`matched_mask_label` is therefore obtained by maximum-overlap one-to-one
matching within each frame. A match is marked `mapping_reliable` when:

- at least 25% of sampled visible skeleton geometry overlaps the assigned mask; and
- its overlap exceeds the next-best mask by at least 0.05.

All candidate overlap scores are preserved in `all_mask_overlap_scores`, so
these audit thresholds can be varied without re-reading the images. Downstream
evaluation must use `matched_mask_label` and must report sensitivity to excluding
or manually resolving `mapping_reliable=False` cases.

## C2 headline counts

- Frames: 10,040
- Pose instances: 11,560
- Reference-mask instances: 11,918
- Nonzero one-to-one geometric matches: 11,286
- Conservative reliable matches: 11,270
- Ambiguous or unmatched pose instances: 290
- Frames with different pose/mask counts: 645
- Pose instances with at least one off-image keypoint: 850
- Frames with at least one off-image keypoint: 826

The SVG audit gallery is generated under `outputs/c2_overlays/index.html` and
contains ordinary controls plus examples of off-image annotations, swapped
pose order, count mismatches, and ambiguous correspondence.
