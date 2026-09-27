#!/usr/bin/env bash
set -euo pipefail

REPO=/nfs/home/talabi/repositories/robust_mis_exp
SAM_SOURCE="${REPO}/.deps/segment-anything"
cd "${REPO}"

if [[ ! -f "${SAM_SOURCE}/segment_anything/__init__.py" ]]; then
  mkdir -p "${REPO}/.deps"
  git clone --depth 1 https://github.com/facebookresearch/segment-anything.git "${SAM_SOURCE}"
fi

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

python3 scripts/run_sam1_pilot.py \
  --checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --segment-anything-path "${SAM_SOURCE}" \
  --output-dir outputs/c6_sam1_vit_h_pilot

python3 scripts/render_c6_gallery.py \
  --results outputs/c6_sam1_vit_h_pilot/per_instance_results.csv
