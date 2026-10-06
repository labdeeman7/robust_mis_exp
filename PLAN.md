# Pose-to-Instance Segmentation on ROBUST-MIPS

## Research question

How often can a frozen promptable segmentation model recover the correct surgical-instrument instance mask when it is given only that instrument's skeletal pose?

The principal result will be instance-mask IoU. We will report the percentage of instrument instances reaching IoU thresholds of 0.50, 0.70, 0.80, and 0.90. An IoU of 0.70--0.80 is expected to be practically useful, but no single organizational default will be imposed by the experiment; stakeholders can select the operating threshold appropriate to their use case.

## Scope and principles

- Use ROBUST-MIPS images, corrected instance masks, and pose annotations directly when the released archive contains all three.
- Do not download the much larger ROBUST-MIS video snippets unless a required image or annotation is absent from ROBUST-MIPS.
- First measure *oracle-pose sufficiency* using ground-truth poses. Evaluate predicted poses only after the prompt encoding is established.
- Treat every instrument as a separate instance. A union-of-instruments score is secondary and must not hide instance-merging failures.
- Keep the pretrained segmentation models frozen for the principal zero-shot experiment.
- Preserve the official procedure-level train/validation/test grouping and avoid frame-level leakage.
- Record exact model checkpoints, package versions, prompts, random seeds, and dataset hashes.

## Checkpoints

### C1 — Acquire and verify ROBUST-MIPS

1. Download the ROBUST-MIPS archive from Synapse entity `syn64023381`.
2. Record its source URL, entity/version identifier, licence, size, and checksum.
3. Confirm whether each selected frame includes:
   - `raw.png`;
   - `instrument_instances.png`;
   - a pose annotation (`toolposes.json` in release v1; called `raw.json` in the paper's schematic).
4. Confirm that there are 10,040 images and reconcile any count discrepancy.
5. Verify that the masks are the ROBUST-MIPS corrected masks, including the stated removal of trocar/cannula instances.
6. Do not acquire ROBUST-MIS video snippets unless this audit proves they are necessary.

Deliverable: a dataset manifest and integrity report. C1 passes when images, corrected masks, poses, splits, licence, and checksums are accounted for.

### C2 — Normalize the dataset without altering source data ✅ Complete (2026-09-27)

1. Keep the downloaded archive and extracted source tree immutable.
2. Build a manifest with one row per image and one row per instrument instance containing:
   - official split, surgery type, procedure ID, and frame ID;
   - image and mask paths;
   - mask instance ID and area;
   - EntryPoint, HingePoint, Tip1, and Tip2 coordinates;
   - visibility state for each keypoint;
   - skeleton edges and transition points;
   - whether any annotated coordinate lies outside the image;
   - number of instruments in the frame.
3. Establish the pose-to-mask instance correspondence. Flag ambiguous or inconsistent examples rather than guessing.
4. Generate visual overlays for a random sample and all detected schema anomalies.

Deliverable: versioned Parquet/CSV manifests, schema documentation, and validation overlays. C2 passes when every usable pose instance maps unambiguously to its reference mask.

C2 outcome: usable equal-count frames were divided into Bucket A (in-frame geometry), Bucket B (one or more coordinates outside the rectangular image), and a manually audited Bucket C. Frames with missing masks or unequal pose/mask counts remain recorded but are excluded from the SAM instance evaluation. Pose-to-mask identity is stored explicitly and never inferred from list order. See `LEARNINGS_JOURNAL.md` and `data/manifests/c2/`.

### C3 — Define the evaluation protocol ✅ Complete (2026-09-27)

Primary unit: instrument instance.

For each matched pose instance, compare the SAM prediction directly with its matched reference-mask label using binary intersection-over-union:

- `IoU = intersection / union`;
- an empty or wrong-instance prediction receives IoU 0;
- report the individual per-instance IoU so results can be re-aggregated later.

Headline summaries:

- mean and median instance IoU;
- number and percentage of failures with IoU < 0.20;
- percentage of instances with IoU >= 0.50;
- percentage of instances with IoU >= 0.70;
- percentage of instances with IoU >= 0.80;
- percentage of instances with IoU >= 0.90.

Report these overall and separately for Buckets A, B, and C. When comparing prompt strategies or SAM checkpoints, use exactly the same matched instances. Keep the complete per-instance result table, including empty predictions and failures, rather than silently filtering them. More elaborate boundary metrics, confidence intervals, and failure taxonomies are optional follow-ups rather than requirements for the first result.

Deliverable: tested metric implementation and a frozen evaluation configuration.

### C4 — Construct a deterministic 200-image pilot ✅ Complete (2026-09-27)

Use 200 distinct images for the first SAM 1 experiment:

- 100 Bucket A frames: 50 single-instrument and 50 multi-instrument;
- 94 Bucket B frames: 47 single-instrument and 47 multi-instrument;
- all six reviewed Bucket C frames.

This produces 318 matched instrument instances. Seed and selections are frozen in `data/manifests/c4_pilot/`.

Deliverable: a seeded pilot manifest with a documented sampling audit.

### C5 — Implement pose-to-prompt encodings ✅ Complete (2026-09-27)

Implement the following model-independent prompt strategies:

1. **P1: Visible keypoints**.
2. **P2: Visible skeleton** — interpolate only edges whose endpoints are both visible and in bounds.
3. **P3: Visible-pose box** — derive a padded box from visible in-bounds keypoints.
4. **P4: Visible keypoints + visible-pose box**.
5. **P5: Visible keypoints + other-instrument visible negatives**.
6. **P6: Visible keypoints + visible-pose box + other-instrument visible negatives**.
7. **P7: Shifted-entry ablation** — when the annotated EntryPoint is outside the image, project its geometry to the image boundary and shift 10 pixels inward. This is explicitly experimental and never part of the default prompt.

Off-image policy:

- never prompt with a keypoint tagged `occluded` or `missing` (the release has no separate `hidden` tag; these are the non-visible states);
- never pass an out-of-bounds coordinate, even in the rare case where it is tagged `visible`;
- do not turn inferred transition or boundary-intersection coordinates into positive point prompts in the primary experiment;
- retain the original visibility/off-image state for stratified analysis.

The pilot contains 2,226 prompts: seven strategies for 318 instances. P7 creates a shifted entry prompt for 97 instances. Prompt generation is deterministic.

Deliverable: prompt JSON plus overlay images for every strategy. C5 passes when unit tests cover missing, occluded, off-image, crossing, and single-keypoint cases.

### C6 — Run the zero-shot SAM 1 prompt-selection pilot ✅ Complete (2026-09-27)

Run the official frozen SAM 1 ViT-H checkpoint on the DGX through Run:ai. Encode each image once and evaluate all seven prompt strategies against the same 318 matched instances.

For SAM's three candidate masks, record:

- the model's highest predicted-quality candidate (deployable rule);
- the best candidate by ground-truth IoU (oracle candidate-selection ceiling, never presented as deployable performance).

Deliverable: reproducible pilot result tables and qualitative panels.

C6 outcome: all 2,226 evaluations completed on an NVIDIA A100-SXM4-40GB. P2 visible skeleton was the strongest overall deployable strategy (mean IoU 0.6628; 66.4% at IoU >= 0.70; 57.5% at IoU >= 0.80). The shifted-entry ablation underperformed unshifted visible points on its applicable subset and is not recommended as the default.

### C7 — Choose the SAM 1 prompt policy ✅ Complete (2026-09-27)

Use the 200-image pilot results to choose the prompt policy before running the full dataset:

1. compare mean/median IoU and success rates at 0.50, 0.70, 0.80, and 0.90;
2. compare the number and percentage of failures with IoU < 0.20;
3. inspect results separately for A, B, and C rather than choosing from the pooled score alone;
4. compare SAM's deployable highest-predicted-quality candidate and the oracle best-of-three ceiling;
5. inspect the best/worst qualitative galleries, especially multi-instrument and crossing cases;
6. decide whether the shifted-entry ablation is beneficial for the applicable off-image subset;
7. select one primary prompt policy, with at most one secondary policy if the bucket-specific trade-off is substantial.

Deliverable: a recorded prompt-selection decision and frozen full-run configuration.

C7 outcome: carry P1 and P2 into the full run. P1 is the minimal visible-keypoint baseline; P2 is the strongest overall pilot policy and densifies only fully visible, in-bounds skeleton edges. Retaining both quantifies whether densification helps at scale without introducing boxes or competitor-dependent negatives.

### C8 — Run P1 and P2 on full Buckets A, B, and C ✅ Complete (2026-09-27)

Run the selected frozen prompt policy over every eligible, explicitly matched instrument instance:

- Bucket A: in-frame-coordinate cases, excluding the six frames assigned to C;
- Bucket B: cases with one or more coordinates outside the rectangular image, excluding the six frames assigned to C;
- Bucket C: the six manually reviewed difficult multi-instrument frames.

The full run must retain the same visibility rule: only explicitly `visible`, in-bounds points may be normal SAM prompts. Report per-instance IoU, IoU < 0.20 failure counts/rates, and the agreed success summaries overall and separately for A, B, and C. Keep all failures and empty predictions. Do not tune the prompt policy after inspecting full-run results.

Deliverable: the main full-dataset feasibility result answering how often pose prompts recover the correct instrument instance.

C8 outcome: all 7,348 frames, 10,525 instruments, and 21,050 P1/P2 evaluations completed successfully. P2 was strongest overall: mean IoU 0.6855, median 0.8434, 15.9% below IoU 0.20, 69.4% at IoU >= 0.70, and 59.2% at IoU >= 0.80. P1 achieved mean IoU 0.6266 with 26.5% below 0.20. P2 reduced the failure count by 1,122 instances, although P1 produced slightly more IoU >= 0.90 masks and achieved one additional IoU >= 0.70 success among the 12 Bucket C instances.

## Optional future paper phase

The current feasibility study ends after C8. If the result supports a paper, a separate plan can cover detailed failure taxonomy, predicted-pose experiments, robustness ablations, confidence intervals, and a publication-grade reproducibility package. These are intentionally not commitments in the current scope. SAM 2 is out of scope for now.

## Planned repository layout

```text
.
├── PLAN.md
├── README.md
├── configs/
├── data/
│   ├── archives/          # ignored; original downloads
│   ├── raw/               # ignored; immutable extracted data
│   ├── manifests/         # generated metadata, where redistribution permits
│   └── README.md
├── notebooks/             # exploration only; production logic lives in src
├── outputs/               # ignored experiment artifacts
├── scripts/
├── src/robust_mips_exp/
└── tests/
```

## Decision log

- **2026-09-25:** Prefer the self-contained ROBUST-MIPS release over ROBUST-MIS because it is reported to include raw images, corrected instance masks, and pose JSON for all 10,040 selected frames.
- **2026-09-27:** Simplify evaluation to per-instance mask IoU and success rates at IoU >= 0.50, 0.70, 0.80, and 0.90. Present 0.70--0.80 as a likely useful range while leaving the default threshold to stakeholders.
- **2026-09-25:** Use ground-truth poses first, followed by predicted-pose and optional temporal experiments.
- **2026-09-25:** Treat other instruments as candidate negative prompts, while explicitly measuring failures when instruments touch or cross.
- **2026-09-25:** The released pose-list order is not a reliable mask-instance identifier. C2 must use explicit one-to-one geometric matching and retain ambiguous/unmatched cases for review; downstream evaluation must never assume `pose_index + 1` identifies the mask.
- **2026-09-27:** C2 completed. SAM point prompts must satisfy both `tag == visible` and rectangular in-bounds checks; coordinates that are occluded, missing, outside the rectangular image, or outside the circular visible field must not be used as positive prompts.
- **2026-09-27:** Use 200 images only to choose the SAM 1 prompt policy, then freeze that policy and run it over all eligible instances in Buckets A, B, and C. Defer SAM 2 and broader publication analyses.
- **2026-09-27:** Test a 10-pixel inward shifted EntryPoint only as an explicit P7 ablation; never silently replace invalid or occluded keypoints in the default prompts.
- **2026-10-06:** Replace the coarse A/B/C reporting split with mutually exclusive per-instrument strata: A = all keypoints visible and in bounds; B = partial pose with occluded/missing keypoints but no off-screen coordinate; C = at least one off-screen coordinate; D = correspondence-hard/manual-review frames. Existing SAM predictions are relabelled and re-summarized without rerunning inference.
