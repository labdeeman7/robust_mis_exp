#!/usr/bin/env bash
set -euo pipefail

JOB_NAME=robust-mips-gallery
runai training standard delete "${JOB_NAME}" -p talabi || true
runai training standard submit "${JOB_NAME}" \
  -p talabi \
  -i aicregistry:5000/talabi:qwen \
  --run-as-user \
  --gpu-devices-request 1 \
  --node-type A100 \
  --host-path path=/nfs/home/talabi,mount=/nfs/home/talabi,readwrite \
  --backoff-limit 0 \
  --command -- bash /nfs/home/talabi/repositories/robust_mis_exp/scripts/gallery_dgx/run.sh
