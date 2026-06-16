#!/usr/bin/env bash
# Fetch results for a madaros-frame-fix SLURM job from OrangeFS.
# Usage: RUN_ID=madaros-frame-fix-<timestamp> ./fetch.sh
#    or: ./fetch.sh madaros-frame-fix-<timestamp>
set -euo pipefail

RUN_ID="${1:-${RUN_ID:-}}"
test -n "${RUN_ID}" || { echo "Usage: $0 <RUN_ID>  or  RUN_ID=... $0" >&2; exit 1; }

NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
ORANGEFS_TMP="${ORANGEFS_TMP:-/orangefs/training/tmp}"

LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }

DEST="slurm-jobs/madaros-frame-fix/results/${RUN_ID}"
mkdir -p "${DEST}"

echo "fetching results from OrangeFS ${ORANGEFS_TMP}/${RUN_ID}.result/ ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
  "tar -czf - '${ORANGEFS_TMP}/${RUN_ID}.result' 2>/dev/null || true" \
  | tar -xzf - -C "${DEST}" --strip-components=4 2>/dev/null || true

echo "fetching slurmout ..."
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- \
  bash -lc "cat '${ORANGEFS_TMP}/${RUN_ID}.slurmout' 2>/dev/null || echo '(not found)'" \
  > "${DEST}/slurmout.txt" 2>/dev/null || true

echo "=== SUMMARY ==="
cat "${DEST}/SUMMARY.txt" 2>/dev/null || echo "(not found yet)"
