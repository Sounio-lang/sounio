#!/usr/bin/env bash
# scripts/ci/dissertation_pbpk28_gpu_parity_gate.sh
#
# GPU parity gate: K-AXI GUM + population-MC kernels vs CPU references.
#
# Cases tracked (device/host bitwise unless noted):
#   gpu-gum-rapamycin .... kernel J-Sigma-JT combine == live ep28_run (5 mg IV, 168 h)
#   gpu-gum-semaglutide .. same, semaglutide 1 mg / 168 h
#   gpu-gum-venlafaxine .. same, venlafaxine XR 75 mg oral / 168 h (NM; oral
#     matrix→gut→blood path + CYP2D6 formation sink, parent-only)
#   gpu-gum-midazolam .... same, midazolam IR 5 mg oral / 168 h (NM; oral
#     gut→blood path dosed via gut0 + CYP3A hepatic sink, parent-only)
#   gpu-gum-haloperidol .. same, haloperidol IR 5 mg oral / 72 h (NM; oral
#     gut→blood path dosed via gut0 + CYP3A4/2D6 hepatic sink, parent-only;
#     BBB/D2 resolved: Kp_brain 15 carries accumulation, D2 occupancy wired
#     as receipt via the Kpuu 3.0 bridge off the parity-checked AUC)
#   gpu-gum-emitter ...... K-AXI asm text + PTX f64 transpile asserts
#   gpu-gum-failclosed ... NaN / negative-variance / zero-variance guards
#   gpu-gum-sabotage .... corrupted combine output MUST be detected (negative control)
#   gpu-gum-batched ...... E-endpoint multi-eval emitter: tid/8 addressing text + PTX asserts
#   gpu-mc-substreams .... substream draws (k=0,3) == MRG32k3a reference, fail-closed index
#   gpu-mc-draws ......... closed-form normal/lognormal spot values
#   gpu-mc-emitter ....... integer advance asm text asserts (no nop fallback)
#   gpu-mc-sim ........... cycle-simulator advance == CPU next_u01 (6 words + z)
#   gpu-mc-sabotage ...... wrong-seed reference MUST mismatch (negative control)
#   gpu-mc-memfed-emitter  one-cubin N-thread advance: get_tid + mem addressing text asserts
#   gpu-mc-memfed-sim .... mem-fed advance == CPU next_u01, tid 0 slice + tid 1 stride
#
# All PENDING drug arms have landed. History: venlafaxine unlocked with the
# oral input path (PK parity in pbpk28-parity cases 10-13); midazolam with
# the oral IR gut0 path and probe-measured CYP3A CL_h (cases 17-19);
# haloperidol with the same IR path, Kp_brain 15 accumulation and the D2
# occupancy receipt via the Kpuu 3.0 bridge (PK parity in cases 14-16).
# This gate never fabricated priors: each arm unlocked on landed ep28 priors.
#
# Sabotage philosophy follows witness_declares_its_sabotage_gate.sh: a check
# that cannot die under mutilation measures nothing. The sabotage arms live
# inside the run-pass tests (self-contained negative controls, no compiler
# switches), and this gate requires their OK lines to be present.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "dissertation_pbpk28_gpu_parity"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_DISSERTATION_PBPK28_GPU_PARITY_DIR:-$(mktemp -d /tmp/sounio-dissertation-pbpk28-gpu-parity.XXXXXX)}"
GUM_LOG="$OUT_DIR/gum.txt"
MC_LOG="$OUT_DIR/mc.txt"

echo "[pbpk28-gpu-parity] out=$OUT_DIR"

echo "[pbpk28-gpu-parity] GUM kernel: $SOUC_BIN run tests/run-pass/gpu_pbpk28_gum_parity.sio"
timeout 900 "$SOUC_BIN" run tests/run-pass/gpu_pbpk28_gum_parity.sio > "$GUM_LOG" 2>&1 || {
  echo "[pbpk28-gpu-parity] FAIL: GUM test returned non-zero" >&2
  tail -n 20 "$GUM_LOG" >&2
  exit 1
}
for marker in "gpu_gum rapamycin OK" "gpu_gum semaglutide OK" "gpu_gum venlafaxine OK" "gpu_gum midazolam OK" "gpu_gum haloperidol OK" "gpu_gum emitter OK" "gpu_gum failclosed OK" "gpu_gum sabotage OK" "gpu_gum batched emitter OK" "PASS gpu_pbpk28_gum_parity"; do
  if ! grep -qF "$marker" "$GUM_LOG"; then
    echo "[pbpk28-gpu-parity] FAIL: GUM log missing marker: $marker" >&2
    tail -n 20 "$GUM_LOG" >&2
    exit 1
  fi
  echo "[pbpk28-gpu-parity] case OK: $marker"
done

echo "[pbpk28-gpu-parity] MC kernel: $SOUC_BIN run tests/run-pass/gpu_pbpk28_mc_parity.sio"
timeout 900 "$SOUC_BIN" run tests/run-pass/gpu_pbpk28_mc_parity.sio > "$MC_LOG" 2>&1 || {
  echo "[pbpk28-gpu-parity] FAIL: MC test returned non-zero" >&2
  tail -n 20 "$MC_LOG" >&2
  exit 1
}
for marker in "gpu_mc substreams OK" "gpu_mc draws OK" "gpu_mc emitter OK" "gpu_mc sim OK" "gpu_mc sabotage OK" "gpu_mc memfed emitter OK" "gpu_mc memfed sim OK" "PASS gpu_pbpk28_mc_parity"; do
  if ! grep -qF "$marker" "$MC_LOG"; then
    echo "[pbpk28-gpu-parity] FAIL: MC log missing marker: $marker" >&2
    tail -n 20 "$MC_LOG" >&2
    exit 1
  fi
  echo "[pbpk28-gpu-parity] case OK: $marker"
done

gate_pass "16/16 GPU parity cases (5 drugs GUM + MC + fail-closed + sabotage + mem-fed MC + batched GUM emitter); all drug arms landed"
