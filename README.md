# ROBUST-MIPS Pose-to-Segmentation Experiments

This repository investigates how accurately frozen promptable segmentation models can convert ROBUST-MIPS surgical-instrument skeletal poses into instance masks.

The research protocol and numbered checkpoints are in [PLAN.md](PLAN.md).

Dataset files are not committed. Acquisition metadata and local layout instructions belong in `data/README.md`.

## Current pilot

The active feasibility study uses SAM 1 ViT-H only. C6 is a 200-image prompt-selection pilot; after C7 freezes the prompt policy, C8 evaluates all eligible instances in Buckets A, B, and C:

```bash
python3 scripts/build_c4_pilot.py
python3 scripts/build_c5_prompts.py
bash scripts/c6_sam1_dgx/submit.sh
```

- C4: 200 distinct images and 318 explicitly matched instrument instances.
- C5: seven deterministic prompt strategies and 2,226 prompts. Normal prompts contain only visible, in-bounds coordinates; P7 is a separately labelled shifted-entry ablation.
- C6: frozen official SAM 1 ViT-H prompt-selection pilot on one Run:ai A100, with per-instance IoU, threshold success rates, saved predictions, and best/worst qualitative galleries.
- C7: select and freeze the primary prompt strategy from the pilot.
- C8: run that strategy over full Buckets A, B, and C without further tuning.

The official checkpoint is stored locally under `checkpoints/` and Meta's source is pinned under `.deps/`; both are ignored by version control.
