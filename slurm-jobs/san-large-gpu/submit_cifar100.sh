#!/usr/bin/env bash
# Submit SAN large architecture training on CIFAR-100 to the Slurm GPU pool.
#
# Usage:
#   bash slurm-jobs/san-large-gpu/submit_cifar100.sh [resnet50|vitlarge|gpt]
#
# Env:
#   SAN_LARGE_ONLY   leg to run (default: resnet50)
#   SAN_LARGE_DELTA_*  (forwarded)
#   SAN_LARGE_TAU_*    (forwarded)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEG="${1:-${SAN_LARGE_ONLY:-resnet50}}"

export SAN_LARGE_DATASET=cifar100
export RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-cifar100-${LEG}}"

bash "${SCRIPT_DIR}/submit.sh" "${LEG}"
