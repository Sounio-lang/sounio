#!/usr/bin/env bash
# Promotion gate: imported exclusive-ref Xoshiro first-draw is trustworthy under
# tip Madaros (2026-08-06 remeasure). lean_single remains the oracle twin.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
PROBE="tests/epistemic_trust/madaros_xoshiro_imported_probe.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== madaros_xoshiro_imported_gate =="
run_eng() {
  local eng="$1" log="$2"
  unset SOUNIO_SOUC_ENGINE || true
  if [[ "$eng" != "madaros" ]]; then
    export SOUNIO_SOUC_ENGINE="$eng"
  fi
  set +e
  "$SOUC" run "$PROBE" >"$log" 2>&1
  local rc=$?
  set -e
  unset SOUNIO_SOUC_ENGINE || true
  [[ "$rc" -eq 0 ]] || { echo "FAIL: engine=$eng rc=$rc"; cat "$log"; exit 1; }
  grep -Fq 'XOSHIRO_IMPORTED PASS' "$log" || {
    echo "FAIL: engine=$eng missing PASS"
    cat "$log"
    exit 1
  }
}

run_eng madaros "$OUT/mad.log"
run_eng lean_single "$OUT/lean.log"
echo "MADAROS_XOSHIRO_IMPORTED_GATE_OK"
