#!/usr/bin/env bash
# Madaros-native qd128_core import smoke (zero-event residual, stdlib leaf).
# Full math::qd128 arithmetic is covered by madaros_qd128_mul_native_v2_gate.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/run-pass/qd128_core_import_native_v2_smoke.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/qdcore.elf"

echo "== madaros_qd128_core_native_v2_gate =="
if ! "$SOUC" compile "$SRC" -o "$ELF" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: compile"
  tail -40 "$OUT/compile.log" || true
  exit 1
fi
chmod +x "$ELF"
LOG="$OUT/run.log"
if ! "$ELF" >"$LOG" 2>&1; then
  echo "FAIL: run"
  cat "$LOG" || true
  exit 1
fi
grep -q 'QD128_CORE_IMPORT_NATIVE_V2 PASS' "$LOG" || {
  echo "FAIL: missing sentinel"
  cat "$LOG" || true
  exit 1
}
echo "MADAROS_QD128_CORE_NATIVE_V2_GATE_OK"
