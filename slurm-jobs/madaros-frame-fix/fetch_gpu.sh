#!/usr/bin/env bash
# Fetch results from the madaros-frame-fix GPU-node SLURM job (submit_gpu.sh).
# Results land on the worker pod that ran the node-pinned job (default: the r770 pod).
# Usage: RUN_ID=madaros-frame-fix-<timestamp> ./fetch_gpu.sh
set -euo pipefail

RUN_ID="${1:-${RUN_ID:-}}"
test -n "${RUN_ID}" || { echo "Usage: $0 <RUN_ID>  or  RUN_ID=... $0" >&2; exit 1; }

NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
WORKER_POD="${WORKER_POD:-slurm-pilot-worker-gpuorangefs-zj2l4}"   # pod on gpuorangefs-r770-proxmox
WORKER_CTR="${WORKER_CTR:-slurmd}"

DEST="slurm-jobs/madaros-frame-fix/results/${RUN_ID}"
mkdir -p "${DEST}"

echo "fetching from ${WORKER_POD}:/tmp/${RUN_ID}.result/ -> ${DEST}/"
${KUBECTL} -n "${NS}" exec "${WORKER_POD}" -c "${WORKER_CTR}" -- \
  tar -czf - "/tmp/${RUN_ID}.result" 2>/dev/null \
  | tar -xzf - -C "${DEST}" --strip-components=3 2>/dev/null || true

echo "fetching slurmout ..."
${KUBECTL} -n "${NS}" exec "${WORKER_POD}" -c "${WORKER_CTR}" -- \
  cat "/tmp/${RUN_ID}.slurmout" > "${DEST}/slurmout.txt" 2>/dev/null || true

echo "=== SUMMARY ==="
cat "${DEST}/SUMMARY.txt" 2>/dev/null || echo "(not found yet — job may still be running)"
