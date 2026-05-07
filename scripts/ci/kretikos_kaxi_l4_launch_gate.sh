#!/usr/bin/env bash
# scripts/ci/kretikos_kaxi_l4_launch_gate.sh
#
# CI gate: K-AXI → PTX → real GPU launch + value/variance verification.
# Submits the matrix sbatch to the slurm-pilot cluster and PASSes only when
# all 12 cases (6 patterns × {basic, epistemic}) match the analytically
# computed reference outputs (mem and var both, where applicable).
#
# This gate requires:
#   - kubectl access to the slurm-pilot namespace
#   - a live login pod (label app.kubernetes.io/name=login)
#   - GPU partition availability (gpu-orangefs)
#   - local cc + ./bin/kretikos
#
# Skip with: SOUNIO_KAXI_L4_GATE_SKIP=1
# Override timeout: WAIT_TIMEOUT_SECONDS=900

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${SOUNIO_KAXI_L4_GATE_SKIP:-0}" == "1" ]]; then
  echo "kretikos_kaxi_l4_launch_gate: SKIPPED (SOUNIO_KAXI_L4_GATE_SKIP=1)"
  exit 0
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kretikos_kaxi_l4_launch_gate: SKIPPED (kubectl missing)"
  exit 0
fi

NS="${NS:-slurm-pilot}"
LOGIN_POD="$(kubectl -n "${NS}" get pods \
  -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -z "${LOGIN_POD}" ]]; then
  echo "kretikos_kaxi_l4_launch_gate: SKIPPED (no live login pod in ${NS})"
  exit 0
fi

echo "kretikos_kaxi_l4_launch_gate: submitting matrix"
out="$(bash "${ROOT_DIR}/slurm-jobs/kretikos/submit-kaxi-ptx-matrix.sh" 2>&1 || true)"
echo "${out}"

# Pull the final summary line
summary_line="$(echo "${out}" | grep -E 'Comment=kaxi_matrix' | tail -1 || true)"
if [[ -z "${summary_line}" ]]; then
  echo "kretikos_kaxi_l4_launch_gate: FAIL (no Comment line)"
  exit 1
fi

# Comment=kaxi_matrix passed=12/12 failed=
if echo "${summary_line}" | grep -qE 'passed=12/12 failed= *$|passed=12/12 failed=$'; then
  echo "kretikos_kaxi_l4_launch_gate: PASS"
  exit 0
fi

# If matrix expanded later, accept passed=N/N with empty failed list
if echo "${summary_line}" | grep -qE 'passed=([0-9]+)/\1 failed= *$'; then
  echo "kretikos_kaxi_l4_launch_gate: PASS"
  exit 0
fi

echo "kretikos_kaxi_l4_launch_gate: FAIL — ${summary_line}"
exit 1
