#!/usr/bin/env bash
# Madaros-native stats::validation import smoke (fixed [f64;256]+n API).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/run-pass/validation_madaros_import_smoke.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/val.elf"

echo "== madaros_validation_import_gate =="
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
grep -q 'VALIDATION_MADAROS_IMPORT PASS' "$LOG" || {
  echo "FAIL: missing sentinel"
  cat "$LOG" || true
  exit 1
}
echo "MADAROS_VALIDATION_IMPORT_GATE_OK"
