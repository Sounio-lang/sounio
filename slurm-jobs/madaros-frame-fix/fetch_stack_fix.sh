#!/usr/bin/env bash
# Fetch results for the stack-fix validation job (no-ulimit proof)
set -euo pipefail

RUN_ID="${1:-madaros-stack-fix-20260617T111142}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
ORANGEFS_TMP="${ORANGEFS_TMP:-/orangefs/training/tmp}"
OUT="${2:-/workspace/sounio/slurm-jobs/madaros-frame-fix/results/${RUN_ID}}"

LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
test -n "${LOGIN_POD}" || { echo "no login pod" >&2; exit 1; }

echo "login pod: ${LOGIN_POD}"

# Check job status
echo "=== SLURM queue ==="
${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "squeue --name=${RUN_ID} 2>/dev/null || true"

echo ""
echo "=== Fetching results from OrangeFS ==="
mkdir -p "${OUT}"

${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
  "ls '${ORANGEFS_TMP}/${RUN_ID}.result/' 2>/dev/null || echo '(result dir not yet created)'"

${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
  "cat '${ORANGEFS_TMP}/${RUN_ID}.result/SUMMARY.txt' 2>/dev/null || echo '(SUMMARY.txt not yet available)'"
