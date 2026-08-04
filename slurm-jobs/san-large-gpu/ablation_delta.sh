#!/usr/bin/env bash
# Threshold ablation driver for SAN large architectures on Slurm GPU.
# Submits one job per delta value for ResNet-50 and ViT-large on CIFAR-10.
#
# Usage:
#   bash slurm-jobs/san-large-gpu/ablation_delta.sh
#
# Env:
#   DRY_RUN=1        print submissions without running submit.sh
#   NS               k8s namespace (forwarded to submit.sh)
#   SBATCH_PARTITION (forwarded to submit.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Threshold grids (Q0.15 confidence scale, same family defaults as the spec).
RESNET_DELTAS=(0.35 0.45 0.55 0.65 0.75)
VIT_DELTAS=(0.25 0.35 0.45 0.55)

submit_one() {
    local leg="$1"
    local delta="$2"
    local var_name="$3"
    local stamp
    stamp="$(date +%Y%m%d-%H%M%S)"
    export RUN_ID="${stamp}-delta${delta}-${leg}"
    export SAN_LARGE_ONLY="${leg}"
    # Clear sibling deltas so the worker uses the family-specific default.
    unset SAN_LARGE_DELTA_RESNET SAN_LARGE_DELTA_VIT SAN_LARGE_DELTA_GPT 2>/dev/null || true
    export "${var_name}=${delta}"

    echo "[ablation] ${leg} ${var_name}=${delta} -> RUN_ID=${RUN_ID}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return
    fi
    bash "${SCRIPT_DIR}/submit.sh" "${leg}"
    # Avoid timestamp collisions across submissions.
    sleep 2
}

for d in "${RESNET_DELTAS[@]}"; do
    submit_one resnet50 "${d}" SAN_LARGE_DELTA_RESNET
done

for d in "${VIT_DELTAS[@]}"; do
    submit_one vitlarge "${d}" SAN_LARGE_DELTA_VIT
done

echo "[ablation] all submissions queued."
