#!/usr/bin/env bash
# Monitor: submit SAN-v3 full-CIFAR-10 when the Kubernetes cluster comes back.
# Usage: bash scripts/dev/submit_san_v3_when_cluster_up.sh
set -euo pipefail

cd /tmp/sounio-san-fpga-blockers-20260804

while true; do
  if kubectl -n slurm-pilot get pods --field-selector=status.phase=Running -o name 2>/dev/null | grep -q slurm-pilot-login; then
    echo "[$(date)] Cluster is up. Submitting SAN-v3..."
    SAN_LARGE_ONLY=resnet50 \
    SAN_LARGE_N_TRAIN=50000 \
    SAN_LARGE_N_VAL=10000 \
    SAN_LARGE_EPOCHS_RESNET=60 \
    SAN_LARGE_TAU_RESNET=0.85 \
    SAN_LARGE_DELTA_RESNET=0.55 \
    SAN_LARGE_SEED=17 \
    SAN_LARGE_V2=1 \
    SAN_LARGE_DATASET=cifar10 \
    SAN_LARGE_BATCH=64 \
    SAN_LARGE_GRAD_ACCUM=2 \
    SAN_LARGE_EVAL_BATCH=512 \
    JOB_TIME=48:00:00 \
    JOB_MEM=128G \
    bash slurm-jobs/san-large-gpu/submit.sh
    echo "[$(date)] Submitted. Exiting monitor."
    break
  else
    echo "[$(date)] Cluster still down. Retrying in 60s..."
    sleep 60
  fi
done
