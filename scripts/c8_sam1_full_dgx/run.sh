#!/usr/bin/env bash
set -euo pipefail

REPO=/nfs/home/talabi/repositories/robust_mis_exp
SAM_SOURCE="${REPO}/.deps/segment-anything"
cd "${REPO}"

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

python3 scripts/run_sam1_pilot.py \
  --prompts data/manifests/c8_full/prompts_p1_p2.jsonl \
  --checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --segment-anything-path "${SAM_SOURCE}" \
  --output-dir outputs/c8_sam1_full_p1_p2

python3 scripts/render_c6_gallery.py \
  --results outputs/c8_sam1_full_p1_p2/per_instance_results.csv \
  --prompts data/manifests/c8_full/prompts_p1_p2.jsonl

python3 scripts/plot_c6_distributions.py \
  --results outputs/c8_sam1_full_p1_p2/per_instance_results.csv
