#!/usr/bin/env bash
# Copilot review (PR #2516), finding 3 on the fifth review round:
# tests/run-pass/madaros_tuple_f64_array_capacity_slot.sio only declares 65
# tuple-array-returning functions with distinct names, which passed even
# under the OLD [i64; 4096]-sized LOWER_FN_TUPLE_ARR_HASH/MASK table (round
# 1's bug) and so cannot detect a regression back to it. This gate GENERATES
# (at run time, never checked in -- same pattern as
# madaros_imported_capacity_gate.sh) a program with 4097 distinct
# tuple-of-f64-arrays-returning functions and destructures the LAST one
# (index 4096, i.e. the 4097th collected), the one entry the old 4096-cap
# would have silently dropped. If LOWER_FN_TUPLE_ARR_* ever truncates again
# at 4096 entries, that function's mask reverts to 0, its destructured
# arrays fall back to the integer path, and the arithmetic below reads the
# wrong values.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_TUPLE_ARR_CAP_GATE_KEEP:-0}"
N=4097
LAST=$((N - 1))

fail() {
  echo "[madaros-tuple-arr-capacity-boundary] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_TUPLE_ARR_CAP_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_TUPLE_ARR_CAP_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-tuple-arr-cap.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_TUPLE_ARR_CAP_GATE_BIN:-$WORK/madaros}"
SRC="$WORK/tuple_arr_cap.sio"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_TUPLE_ARR_CAP_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "Madaros is missing or not executable: $MADAROS_ELF"

# Every function is DISTINCT (not a fixed handful repeated), so this also
# exercises the round-2 collision-safety fix (two independent hashes) at
# scale, not just the raw entry count.
for i in $(seq 0 "$LAST"); do
  printf 'fn tarr%s() -> ([f64; 2], [f64; 2]) with Mut, Panic { var a: [f64; 2] = [0.0; 2]; var b: [f64; 2] = [0.0; 2]; a[0] = %s.0; b[0] = 1.0; (a, b) }\n' \
    "$i" "$i" >>"$SRC"
done

{
  echo 'fn main() -> i32 with IO, Mut, Panic {'
  echo "    let (oa, ob) = tarr${LAST}()"
  echo '    let s0: f64 = oa[0] * 2.0'
  echo '    let s1: f64 = 2.0 * ob[0]'
  echo "    if s0 == $((LAST * 2)).0 && s1 == 2.0 {"
  echo '        println("TUPLE_ARR_CAPACITY_BOUNDARY PASS")'
  echo '        return 0'
  echo '    }'
  echo '    1'
  echo '}'
} >>"$SRC"

set +e
RUN_LOG="$WORK/run.log"
STACK_KB="${SOUNIO_MADAROS_TUPLE_ARR_CAP_GATE_STACK_KB:-524288}"
(
  # Same treatment as madaros_imported_capacity_gate.sh: the raw ELF SEGVs
  # (rc=139) on a program this size (4097 functions) under the shell's
  # default stack limit.
  if [[ "$STACK_KB" == "0" ]]; then
    ulimit -s unlimited 2>/dev/null || true
  else
    ulimit -s "$STACK_KB" 2>/dev/null || true
  fi
  exec timeout 300 "$MADAROS_ELF" run "$SRC"
) >"$RUN_LOG" 2>&1
run_rc=$?
set -e

if [[ "$run_rc" != "0" ]]; then
  tail -n 60 "$RUN_LOG" >&2
  fail "run exited rc=$run_rc for a $N-function tuple-array capacity witness"
fi
grep -Fxq 'TUPLE_ARR_CAPACITY_BOUNDARY PASS' "$RUN_LOG" || {
  tail -n 60 "$RUN_LOG" >&2
  fail "exact PASS marker missing -- entry $LAST (past the old 4096-entry cap) did not classify as an f64 array"
}

echo "[madaros-tuple-arr-capacity-boundary] PASS: entry $LAST of $N distinct tuple-array-returning functions still classified correctly"
