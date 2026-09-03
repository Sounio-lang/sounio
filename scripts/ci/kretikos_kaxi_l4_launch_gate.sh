#!/usr/bin/env bash
# scripts/ci/kretikos_kaxi_l4_launch_gate.sh
#
# CI gate: K-AXI → PTX → real GPU launch + value/variance verification.
#
# Execution paths (in priority order):
#   1. RUN_LOCAL=1 or local GPU detected (A5000): run directly on local GPU
#      via submit-kaxi-ptx-matrix.sh RUN_LOCAL=1.  Local GPU always wins.
#   2. Slurm cluster (kubectl + live login pod): fallback when no local GPU.
#
# Skip with: SOUNIO_KAXI_L4_GATE_SKIP=1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${SOUNIO_KAXI_L4_GATE_SKIP:-0}" == "1" ]]; then
  echo "kretikos_kaxi_l4_launch_gate: SKIPPED (SOUNIO_KAXI_L4_GATE_SKIP=1)"
  exit 0
fi

# Detect local GPU (nvidia-smi succeeds) for the local-run path.
_has_local_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

# --- Local GPU path (preferred) ---
# Local GPU always wins: avoids Slurm scheduling delays and node-unavail failures.
_try_local=0
if [[ "${RUN_LOCAL:-0}" == "1" ]]; then
  _try_local=1
elif _has_local_gpu; then
  _try_local=1
fi

if [[ "${_try_local}" == "1" ]]; then
  echo "kretikos_kaxi_l4_launch_gate: running matrix on local GPU"
  out="$(RUN_LOCAL=1 bash "${ROOT_DIR}/slurm-jobs/kretikos/submit-kaxi-ptx-matrix.sh" 2>&1 || true)"
  echo "${out}"
  summary_line="$(echo "${out}" | grep -E '^kaxi_matrix' | tail -1 || true)"
  if [[ -z "${summary_line}" ]]; then
    echo "kretikos_kaxi_l4_launch_gate: FAIL (no kaxi_matrix summary line)"
    exit 1
  fi
  if echo "${summary_line}" | grep -qE 'passed=([0-9]+)/\1 failed= *$'; then
    echo "kretikos_kaxi_l4_launch_gate: PASS (local GPU)"
    exit 0
  fi
  echo "kretikos_kaxi_l4_launch_gate: FAIL -- ${summary_line}"
  exit 1
fi

# --- Slurm cluster path ---
if ! command -v kubectl >/dev/null 2>&1; then
  echo "kretikos_kaxi_l4_launch_gate: SKIPPED (kubectl missing, no local GPU)"
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

echo "kretikos_kaxi_l4_launch_gate: submitting matrix to Slurm"
out="$(bash "${ROOT_DIR}/slurm-jobs/kretikos/submit-kaxi-ptx-matrix.sh" 2>&1 || true)"
echo "${out}"

summary_line="$(echo "${out}" | grep -E 'Comment=kaxi_matrix' | tail -1 || true)"
if [[ -z "${summary_line}" ]]; then
  echo "kretikos_kaxi_l4_launch_gate: FAIL (no Comment line)"
  exit 1
fi

if echo "${summary_line}" | grep -qE 'passed=([0-9]+)/\1 failed= *$'; then
  echo "kretikos_kaxi_l4_launch_gate: PASS"
  exit 0
fi

echo "kretikos_kaxi_l4_launch_gate: FAIL -- ${summary_line}"
exit 1
