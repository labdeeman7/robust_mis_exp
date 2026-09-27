#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${JOB_NAME:-robust-mips-sam1-full-p1-p2}"

runai training standard delete "${JOB_NAME}" -p talabi || true
runai training standard submit "${JOB_NAME}" \
  -p talabi \
  -i aicregistry:5000/talabi:qwen \
  --run-as-user \
  --gpu-devices-request 1 \
  --node-type A100 \
  --large-shm \
  --host-ipc \
  --host-path path=/nfs/home/talabi,mount=/nfs/home/talabi,readwrite \
  --backoff-limit 0 \
  --command -- bash /nfs/home/talabi/repositories/robust_mis_exp/scripts/c8_sam1_full_dgx/run.sh
