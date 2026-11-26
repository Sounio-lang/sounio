#!/usr/bin/env bash
# A type error in an imported module fails the lean_single build.
#
# Before this gate, tc_mark_failed and note_limit_error returned early for
# functions from imported modules: the seed printed `error: ...`, exited 0 and
# wrote the ELF anyway. tests/compiler/lean_single_import_error_fatal/refused
# is the witness: the seed refuses `saved.2` in tupmod/lib.sio, and the ELF it
# used to write printed 1547925990675276 instead of 1234. Now it must exit
# nonzero, print `typecheck: failed` and write no output file. The positive
# control, clean/, computes the same value through a struct and must still
# build and print 1234.
#
#   scripts/ci/lean_single_import_error_fatal_gate.sh
#   SOUNIO_LEAN_SINGLE_BIN=<elf>  test another lean_single compiler
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${SOUNIO_LEAN_SINGLE_BIN:-$ROOT/bin/souc-lean-single-x86_64}"
FIX="$ROOT/tests/compiler/lean_single_import_error_fatal"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

fail() { echo "[lean-single-import-error-fatal] FAIL: $*" >&2; exit 1; }
say() { echo "[lean-single-import-error-fatal] $*"; }

unset SOUC_BIN SOUNIO_SOUC_BIN
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
ulimit -s unlimited 2>/dev/null || true
[[ -x "$SEED" ]] || fail "missing lean_single compiler $SEED"

# Negative: the error is in the imported module, not in main.sio.
rc=0
(cd "$FIX/refused" && "$SEED" main.sio "$W/refused.elf") >"$W/refused.log" 2>&1 || rc=$?
grep -q '^error: tuple index out of bounds at \./tupmod/lib\.sio:' "$W/refused.log" \
  || { tail -20 "$W/refused.log" >&2; fail "refused/: no 'tuple index out of bounds' error in tupmod/lib.sio"; }
[[ $rc -ne 0 ]] || fail "refused/: the seed printed an error in an imported module and exited 0"
grep -Fxq 'typecheck: failed' "$W/refused.log" || fail "refused/: rc $rc without 'typecheck: failed'"
[[ ! -e "$W/refused.elf" ]] || fail "refused/: rc $rc but an output file was written"
say "refused: rc $rc, typecheck: failed, no output file"

# Positive control: a clean import compiles and runs.
rc=0
(cd "$FIX/clean" && "$SEED" main.sio "$W/clean.elf") >"$W/clean.log" 2>&1 || rc=$?
[[ $rc -eq 0 ]] || { tail -20 "$W/clean.log" >&2; fail "clean/: rc $rc"; }
! grep -q '^error' "$W/clean.log" || fail "clean/: the seed printed an error"
chmod +x "$W/clean.elf"
out="$("$W/clean.elf")" || fail "clean/: the program exited rc $?"
[[ "$out" == "1234" ]] || fail "clean/: printed '$out', expected 1234"
say "clean: rc 0, prints $out"

echo "LEAN_SINGLE_IMPORT_ERROR_FATAL_OK"
