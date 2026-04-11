#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PREPARE_SCRIPT="${ROOT_DIR}/scripts/gpu/prepare_brain_ossm_snapshot.sh"

if [[ ! -x "${PREPARE_SCRIPT}" ]]; then
  echo "missing helper script: ${PREPARE_SCRIPT}" >&2
  exit 1
fi

SNAPSHOT_DIR="$("${PREPARE_SCRIPT}" fractal20)"
RUN_ID="${RUN_ID:-brain-ossm-fractal20-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_DIR="${ORANGEFS_RESULTS_DIR:-/orangefs/training/sounio/ossm-results/fractal20}"
PAYLOAD_TARBALL="${LOCAL_TARBALL:-/tmp/${RUN_ID}.tgz}"

rm -f "${PAYLOAD_TARBALL}"
tar -C "${SNAPSHOT_DIR}" -czf "${PAYLOAD_TARBALL}" .
tar -tzf "${PAYLOAD_TARBALL}" >/dev/null

echo "Prepared 20-seed Fractal-G2 snapshot at ${SNAPSHOT_DIR}"
echo "Submitting with RUN_ID=${RUN_ID}"
echo "Stable results directory: ${RESULTS_DIR}"

SOUNIO_DIR="${SNAPSHOT_DIR}" \
LOCAL_SNAPSHOT_DIR="${SNAPSHOT_DIR}" \
LOCAL_TARBALL="${PAYLOAD_TARBALL}" \
RUN_ID="${RUN_ID}" \
ORANGEFS_RESULTS_DIR="${RESULTS_DIR}" \
FORCE_RESTAGE="${FORCE_RESTAGE:-1}" \
bash "${ROOT_DIR}/slurm-jobs/brain-ossm/submit-fractal-g2-gpu.sh"
