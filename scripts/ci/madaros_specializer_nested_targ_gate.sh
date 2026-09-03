#!/usr/bin/env bash
# Specializer: nested turbofish type arguments must not collide.
#
# Guards BLK-20260903-specializer-nested-targ-collision. Two instantiations of
# one generic template whose type arguments are themselves generic used to
# mangle to the same name, hash equal, and share a single specialization —
# without tripping the second-distinct-instantiation poison guard.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/run-pass/specializer_nested_targ_distinct.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/nested_targ.elf"

echo "== madaros_specializer_nested_targ_gate =="

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

grep -q 'SPECIALIZER_NESTED_TARG_OK' "$LOG" || {
  echo "FAIL: missing sentinel"
  cat "$LOG" || true
  exit 1
}

echo "MADAROS_SPECIALIZER_NESTED_TARG_GATE_OK"
