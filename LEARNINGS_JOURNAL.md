# ROBUST-MIPS learnings journal

This is a living record of dataset facts, decisions, surprises, and implementation lessons. Add entries as the work progresses; do not rewrite inconvenient findings away.

## 2026-09-27 — Acquisition and structure

- ROBUST-MIPS release v1 contains 10,040 frame directories, each with `raw.png`, `instrument_instances.png`, and `toolposes.json`.
- The corrected instance masks are already included, so the larger ROBUST-MIS video download is not needed for the principal still-image experiment.
- Instance masks are 8-bit single-channel PNGs, not LabelMe JSON. Pixel value 0 is background; nonzero values such as 50, 100, and 150 identify instrument instances. These are instance IDs, not semantic instrument classes.
- Pose JSON and mask PNG are separate annotations. The pose list contains `nodes`, visibility `tags`, `edges`, and `transitions`, but no explicit foreign key to a mask label.
- Pose-list order must not be treated as mask identity. Correspondence is stored explicitly as `(frame_key, pose_index, mask_label)`.

## 2026-09-27 — Pose-to-mask correspondence

- Geometry can establish most pose-to-mask matches. The useful score is skeleton containment in a dilated mask: the fraction of in-frame skeleton centreline lying within a 20-pixel dilation of a candidate mask. Unlike ordinary IoU, complete containment can score 1.0 without penalizing the mask for being wider than the skeleton.
- Matching is a one-to-one assignment within each frame, not an independent nearest-mask decision for every pose.
- Six initially suspicious multi-instrument examples were colour-reviewed. After fixing an ambiguous visualization convention, the human review and 20-pixel dilation matcher agreed on all 12 pose instances.
- Visualization colours must encode correspondence, not unrelated pose-list and mask-list indices. A pose and its assigned mask now always share the same colour.
- Frames with unequal pose and mask counts, poses without masks, or masks without poses remain in the audit manifests but cannot support the primary pose-to-instance SAM evaluation.
- Bucket A contains equal-count, geometrically matched frames without coordinates outside the rectangular image.
- Bucket B contains equal-count, geometrically matched frames with at least one coordinate outside the rectangular image.
- Bucket C contains the explicitly reviewed difficult multi-instrument cases.

## 2026-09-27 — Visibility and prompt safety

- The release uses three visibility tags: `visible`, `occluded`, and `missing`. It does not use a distinct literal `hidden` tag.
- Coordinate bounds and visibility are independent. A keypoint can lie inside the rectangular image while being outside the circular endoscopic field of view and correctly tagged `occluded`.
- Example: `Stage_3/Sigmoid/8/33000`, pose 1 has in-bounds EntryPoint and HingePoint coordinates tagged `occluded`; only its two visible tips are valid SAM point prompts.
- Across the release, 1,666 of 1,673 out-of-bounds coordinates are tagged `occluded`; seven are tagged `visible`. Bounds therefore cannot replace visibility tags, and visibility tags cannot replace bounds checks.
- Primary SAM prompt rule: use a keypoint only when `tag == visible` and `0 <= x < width` and `0 <= y < height`.
- Never use `occluded`, `missing`, out-of-bounds, or outside-visible-field points as positive SAM prompts. A transition or inferred boundary point is not a primary positive prompt either.
- Occluded or off-image geometry may still help establish annotation correspondence. Matching-time geometry and model prompt inputs are separate concepts and must not be conflated.

## 2026-09-27 — Evaluation agreement for C3

- The evaluation unit is one matched instrument instance, not the union of all instruments in a frame.
- Compare each predicted binary mask with its matched reference instance using IoU.
- Retain every result, including empty masks and wrong-instance selections; these receive IoU 0 rather than being filtered out.
- Report mean and median IoU plus the percentage of instances reaching IoU thresholds 0.50, 0.70, 0.80, and 0.90.
- An IoU threshold in the 0.70--0.80 range is expected to represent a useful result, but stakeholders will choose the default appropriate to their application.
- Report results overall and separately for Buckets A, B, and C so off-frame and difficult correspondence cases cannot be hidden by the aggregate.
- The original A/B/C split was too coarse: old A only meant no off-image coordinate, so it still contained many occluded or missing keypoints. On 2026-10-06 we replaced it with per-instrument A/B/C/D strata: A complete and visible; B partial but in-frame; C off-screen; D correspondence-hard. The full evaluation contains 2,946/6,801/766/12 instances respectively. This is a reporting-only correction; retained SAM predictions and IoUs were not recomputed.

## Open questions

- Which visible-point combination gives SAM the best trade-off: all visible keypoints, tips only, sparse skeleton samples, or a prompt box?
- How should a pose with only one visible in-frame keypoint be prompted?
- Do negative points from other instruments improve separation when instruments touch or cross?
- Does the explicit 10-pixel inward shifted-entry ablation improve SAM 1 for off-image EntryPoints, or does it introduce misleading positive prompts?
- Should later experiments use transition points for boxes or correspondence while continuing to exclude them from primary positive point prompts?

## 2026-09-27 — Pilot implementation

- The 200-image C6 run is a prompt-selection pilot, not the final dataset evaluation.
- The deterministic pilot has 200 distinct images and 318 matched instrument instances: 100/155 from A, 94/151 from B, and 6/12 from C.
- A and B are balanced between single- and multi-instrument frames; all six manually reviewed C frames are included and excluded from the A/B draw.
- Seven prompt policies yield 2,226 prompts. The six normal policies use only visible, in-bounds points. The seventh is the isolated shifted-entry ablation and applies to 97 instances.
- The C6 baseline is official frozen SAM 1 ViT-H, run on an A100 through Run:ai. SAM's predicted-quality selection is the deployable result; best-of-three ground-truth candidate selection is recorded only as an oracle ceiling.
- After choosing and freezing the prompt policy from the pilot, run it over every eligible instance in full Buckets A, B, and C. Do not revise the policy based on the full-run results.
- SAM 2 and broader publication-scale analyses remain deferred; the full SAM 1 A/B/C run is part of the current feasibility study.

## 2026-09-27 — C6 SAM 1 pilot result

- The Run:ai job completed successfully on an NVIDIA A100-SXM4-40GB: 200 images, 318 instances, seven strategies, and 2,226 evaluated prompts.
- P2 visible skeleton was strongest overall by deployable candidate selection: mean IoU 0.6628, median 0.8438, 71.7% at IoU >= 0.50, 66.4% at IoU >= 0.70, 57.5% at IoU >= 0.80, and 23.9% at IoU >= 0.90.
- P2 also had the best mean IoU in B (0.5760) and C (0.4706). In A, P4 visible points plus box was slightly higher (0.7859) than P2 (0.7623).
- Adding other-instrument negative keypoints provided little overall benefit over visible points alone and did not beat P2.
- Boxes helped substantially in A but harmed B and C, likely because a box derived from sparse visible points poorly represents truncated/off-frame instruments.
- On the 97 instances where P7 generated a shifted EntryPoint, P1 mean IoU was 0.5338 and P7 mean IoU was 0.5191. Shifting helped 51 individual cases and hurt 46, but its average effect was negative; it should not be the default.
- Candidate selection remains a major limitation for point-based prompts: P2's deployable mean IoU was 0.6628 versus an oracle best-of-three mean of 0.7856.
- The initial assumption that pose-to-instance-mask conversion would be trivial with an off-the-shelf promptable segmenter is not supported. Even the strongest zero-shot policy has an 18.2% catastrophic-failure rate at IoU < 0.20, and difficult Bucket C remains unreliable.
- The evidence currently supports this interpretation: pose is useful segmentation guidance, but zero-shot SAM 1 does not reliably resolve missing appearance information, crossings, occlusion, and candidate-mask selection. Fine-tuning SAM's mask decoder or a similarly targeted adaptation is likely necessary for dependable performance; this remains a hypothesis to test rather than a demonstrated conclusion.
- P1 and P2 are the clearest prompt policies to carry forward: P1 is the minimal visible-pose baseline, while P2 adds dense positive points along visible straight-line skeleton edges. P2 is passed to SAM as a collection of positive point prompts, not as a mask prompt.
- P2 points are target-pose-derived, but they are not guaranteed to lie on visible target pixels: a straight annotated edge can traverse background, an occluded section, or another instrument. This is especially important in Bucket C.

## Prompt ideas for future experiments

- SAM 2.1 image prediction with exactly the same P1/P2 protocol, keeping temporal propagation separate unless corresponding video clips are available.
- Fine-tune only the SAM/SAM 2 mask decoder first; compare with LoRA or limited encoder adaptation if decoder-only tuning is insufficient.
- Rasterize the pose as a narrow shaft/jaw prior and test it as a mask prompt or an additional learned input instead of representing the entire skeleton as equal positive points.
- Use dense negative points along competing instruments, rather than only their annotated visible keypoints.
- Add local negative rings around ambiguous skeleton segments to discourage masks from expanding into adjacent tissue or instruments.
- Prompt shaft and jaws separately, then merge or select component masks with a pose-consistency score.
- Use iterative correction: run P1/P2, detect prediction regions inconsistent with the target pose or overlapping competitors, and add deterministic corrective positives/negatives.
- Learn a candidate-mask selector. The large deployable-to-oracle gap shows that SAM frequently generates a good mask but ranks the wrong candidate highest.
- Derive oriented or piecewise boxes from visible shaft segments instead of one axis-aligned box spanning sparse points.

## 2026-09-27 — Why SAM 2 is deferred

- ROBUST-MIPS provides sparse annotated frames at approximately 30-second intervals, not a contiguous pose/video sequence suitable for temporal propagation.
- SAM 2 could still be tested as a static-image foundation model, but its temporal memory would not address this dataset's missing information under the present release structure.
- For the current feasibility question, the more direct next step is therefore the full-dataset SAM 1 comparison of P1 and P2. SAM 2 remains a possible future static-image baseline, not the default current experiment.

## 2026-09-27 — Full P1/P2 evaluation set

- The mutually exclusive full set contains 7,348 frames and 10,525 matched instruments: A has 6,595/9,178 frames/instances, B has 747/1,335, and C has 6/12.
- Correspondence sources are 10,344 high-confidence C2 geometric assignments, 169 assignments recomputed using 20-pixel dilated-skeleton coverage, and 12 manually reviewed C assignments.
- P1 and P2 produce 21,050 evaluations. One instrument has no visible in-bounds point under either policy; it is retained as an explicit no-valid-prompt failure with an empty prediction and IoU 0 rather than being excluded or assigned an invented prompt.

## 2026-09-27 — Full P1/P2 results

- The full Run:ai job completed successfully on an A100: 7,348 frames, 10,525 instruments, and 21,050 evaluated prompts, with one prediction retained for every evaluation.
- P1 overall: mean IoU 0.6266, median 0.8418, 2,791/10,525 (26.5%) below IoU 0.20, 64.2% at IoU >= 0.70, and 56.9% at IoU >= 0.80.
- P2 overall: mean IoU 0.6855, median 0.8434, 1,669/10,525 (15.9%) below IoU 0.20, 69.4% at IoU >= 0.70, and 59.2% at IoU >= 0.80.
- P2 reduced catastrophic failures by 1,122 instances relative to P1 and is the stronger general policy.
- P1 retained a slightly higher IoU >= 0.90 rate (29.2% versus 26.8%), showing that densification mainly prevents severe failures rather than improving the best masks.
- Bucket A remains substantially easier: P2 mean IoU 0.7004 and 14.3% failures. Bucket B remains difficult: P2 mean 0.5851 and 26.4% failures. Bucket C is too small for a stable ranking and remains unreliable: P2 mean 0.4706 with 4/12 failures; P1 has one more instance at IoU >= 0.70.
- Candidate selection is still a major opportunity: P2 deployable mean IoU is 0.6855 versus an oracle best-of-three mean of 0.8077.
- The full result reinforces the central conclusion: visible pose is useful and dense visible skeleton prompting materially improves reliability, but zero-shot SAM 1 is not dependable enough to make pose-to-instance-mask conversion trivial.
