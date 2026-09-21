#!/usr/bin/env bash
# Madaros D6 closeout: module-level const from non-main local fn.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/epistemic_trust/madaros_d6_const_nonmain.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/d6.elf"

echo "== madaros_d6_const_nonmain_gate =="
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
grep -q 'D6_CONST_NONMAIN_OK' "$LOG" || {
  echo "FAIL: missing sentinel"
  cat "$LOG" || true
  exit 1
}
echo "MADAROS_D6_CONST_NONMAIN_GATE_OK"
