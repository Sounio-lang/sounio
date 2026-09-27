#!/usr/bin/env bash
# Madaros-native math::qd128::qd_mul import smoke (stdlib reshape: nine-sum via [f64;9]).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/run-pass/qd128_mul_native_v2_smoke.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/qdmul.elf"

echo "== madaros_qd128_mul_native_v2_gate =="
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
grep -q 'QD128_MUL_NATIVE_V2 PASS' "$LOG" || {
  echo "FAIL: missing sentinel"
  cat "$LOG" || true
  exit 1
}
echo "MADAROS_QD128_MUL_NATIVE_V2_GATE_OK"
