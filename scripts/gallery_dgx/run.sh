#!/usr/bin/env bash
set -euo pipefail

REPO=/nfs/home/talabi/repositories/robust_mis_exp
cd "${REPO}"

python3 scripts/render_c6_gallery.py \
  --results outputs/c8_sam1_full_p1_p2/per_instance_results.csv \
  --prompts data/manifests/c8_full/prompts_p1_p2.jsonl \
  --by-bucket \
  --per-tail 3
