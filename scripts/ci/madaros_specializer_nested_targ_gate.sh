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

# The multi-instantiation shapes. They used to be refused ("more than one
# type-argument list in a single compilation unit") because a second distinct
# instantiation poisoned the template back to its unspecialized body, and a
# struct-typed argument then came back as zero. Every distinct instantiation now
# gets its own clone, so both must compile, run, and print the values: the
# struct-argument one is still the load-bearing case, since the poisoned path
# compiled it with rc=0 and printed zeros.
for shape in \
  specializer_multi_instantiation_struct_args:SPECIALIZER_MULTI_INSTANTIATION_STRUCT_ARGS_OK \
  specializer_multi_instantiation_nonscalar:SPECIALIZER_MULTI_INSTANTIATION_NONSCALAR_OK
do
  name="${shape%%:*}"
  sentinel="${shape#*:}"
  src="tests/run-pass/$name.sio"
  melf="$OUT/$name.elf"
  mlog="$OUT/$name.log"
  if ! "$SOUC" compile "$src" -o "$melf" >"$mlog" 2>&1; then
    echo "FAIL: $src did not compile"
    tail -40 "$mlog" || true
    exit 1
  fi
  chmod +x "$melf"
  if ! "$melf" >"$mlog.run" 2>&1; then
    echo "FAIL: $src exited non-zero"
    cat "$mlog.run" || true
    exit 1
  fi
  grep -q "$sentinel" "$mlog.run" || {
    echo "FAIL: $src missing $sentinel"
    cat "$mlog.run" || true
    exit 1
  }
done

echo "MADAROS_SPECIALIZER_NESTED_TARG_GATE_OK"
