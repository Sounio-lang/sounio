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

# The refused shapes. The struct-argument one is the load-bearing case: on the
# compiler before this guard it compiled with rc=0 and printed zeros, so a gate
# that only checked the nested shape would not have caught the wrong code.
REFUSAL='more than one type-argument list in a single compilation unit'
for refused in \
  tests/compile-fail/specializer_multi_instantiation_struct_args.sio \
  tests/compile-fail/specializer_multi_instantiation_nonscalar.sio
do
  RLOG="$OUT/$(basename "$refused" .sio).log"
  if "$SOUC" compile "$refused" -o "$OUT/refused.elf" >"$RLOG" 2>&1; then
    echo "FAIL: $refused was accepted"
    tail -40 "$RLOG" || true
    exit 1
  fi
  grep -qF "$REFUSAL" "$RLOG" || {
    echo "FAIL: $refused rejected for the wrong reason"
    tail -40 "$RLOG" || true
    exit 1
  }
done

echo "MADAROS_SPECIALIZER_NESTED_TARG_GATE_OK"
