#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 /tmp/moonshot-a-phase-status.<id>" >&2
  exit 2
fi

PREV_DIR="$1"
SCALAR="$PREV_DIR/slurm_scalar/kretikos_kaxi_sinkhorn16_slurm_gate.v1.json"
EPI="$PREV_DIR/slurm_f32_epistemic/kretikos_kaxi_sinkhorn16_epistemic_slurm_gate.v1.json"

if [[ ! -f "$SCALAR" ]]; then
  echo "moonshot_a_phase_status_adopt_runtime_artifacts: missing scalar artifact: $SCALAR" >&2
  exit 1
fi
if [[ ! -f "$EPI" ]]; then
  echo "moonshot_a_phase_status_adopt_runtime_artifacts: missing f32-epistemic artifact: $EPI" >&2
  exit 1
fi

OUT_DIR="${SOUNIO_MOONSHOT_A_PHASE_STATUS_DIR:-$(mktemp -d /tmp/moonshot-a-phase-status-adopt.XXXXXX)}"

SOUNIO_MOONSHOT_A_SCALAR_SLURM_ARTIFACT="$SCALAR" \
SOUNIO_MOONSHOT_A_EPI_SLURM_ARTIFACT="$EPI" \
SOUNIO_MOONSHOT_A_PHASE_STATUS_DIR="$OUT_DIR" \
  bash scripts/ci/moonshot_a_phase_status_gate.sh

python3 scripts/ci/moonshot_a_phase_status_verify.py "$OUT_DIR/moonshot_a_phase_status.v1.json"
echo "moonshot_a_phase_status_adopt_runtime_artifacts: out=$OUT_DIR source=$PREV_DIR"
