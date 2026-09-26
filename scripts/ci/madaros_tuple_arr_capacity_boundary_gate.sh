#!/usr/bin/env bash
# Copilot review (PR #2516), finding 3 on the fifth review round:
# tests/run-pass/madaros_tuple_f64_array_capacity_slot.sio only declares 65
# tuple-array-returning functions with distinct names, which passed even
# under the OLD [i64; 4096]-sized LOWER_FN_TUPLE_ARR_HASH/MASK table (round
# 1's bug) and so cannot detect a regression back to it. This gate GENERATES
# (at run time, never checked in -- same pattern as
# madaros_imported_capacity_gate.sh) a program with 4097 distinct
# tuple-of-f64-arrays-returning functions and verifies the LAST one
# (index 4096, i.e. the 4097th collected), the one entry the old 4096-cap
# would have silently dropped.
#
# Round 6, finding 3 (Copilot again): the FIRST version of this gate called
# only tarr4096 from `main` and compiled the program through the normal
# `madaros run` pipeline. Round 5's own finding-A fix -- filtering the
# collector's input through spec_dce_filter_with_global_marks before
# preregistration -- means the other 4096 generated-but-unreachable
# functions never reach lower_fn_tuple_f64_arrays_collect at all under that
# pipeline: DCE prunes them first, so the gate only ever populated ONE
# table entry and could not have caught a regression back to the old
# 4096-entry cap. Making all 4097 functions genuinely reachable from `main`
# risks a different capacity wall (IR_MAX_INSTRS, ~16384, via 4097 call
# sites in one function body), so instead this uses the
# --probe-tuple-arr-capacity CLI flag (compiler/main.sio), which calls
# lower_fn_tuple_f64_arrays_collect directly on the RAW, un-DCE'd parsed
# item list -- bypassing the whole module_frontend pipeline, DCE included --
# and reports one function's resulting mask. That populates the table with
# every one of the 4097 generated functions, matching what this gate always
# meant to test.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_TUPLE_ARR_CAP_GATE_KEEP:-0}"
N=4097
LAST=$((N - 1))
TARGET_FN="tarr${LAST}"

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
# exercises the collision-safety fix -- exact interned-name identity via
# ir_intern_name, not a hash comparison (self-hosted/ir/lower.sio, see
# LOWER_FN_TUPLE_ARR_NAME_ID's own comment) -- at scale, not just the raw
# entry count. None of these need to be reachable from `main` -- the probe
# collects straight off the parsed item list.
for i in $(seq 0 "$LAST"); do
  printf 'fn tarr%s() -> ([f64; 2], [f64; 2]) with Mut, Panic { var a: [f64; 2] = [0.0; 2]; var b: [f64; 2] = [0.0; 2]; a[0] = %s.0; b[0] = 1.0; (a, b) }\n' \
    "$i" "$i" >>"$SRC"
done
echo 'fn main() -> i32 with IO { 0 }' >>"$SRC"

set +e
RUN_LOG="$WORK/run.log"
STACK_KB="${SOUNIO_MADAROS_TUPLE_ARR_CAP_GATE_STACK_KB:-524288}"
(
  if [[ "$STACK_KB" == "0" ]]; then
    ulimit -s unlimited 2>/dev/null || true
  else
    ulimit -s "$STACK_KB" 2>/dev/null || true
  fi
  exec timeout 300 "$MADAROS_ELF" --probe-tuple-arr-capacity "$SRC" "$TARGET_FN"
) >"$RUN_LOG" 2>&1
run_rc=$?
set -e

if [[ "$run_rc" != "0" ]]; then
  tail -n 60 "$RUN_LOG" >&2
  fail "--probe-tuple-arr-capacity exited rc=$run_rc for a $N-function tuple-array capacity witness"
fi

MASK_LINE="$(grep -E "^probe_tuple_arr_capacity: fn=${TARGET_FN} mask=[0-9]+\$" "$RUN_LOG" || true)"
if [[ -z "$MASK_LINE" ]]; then
  tail -n 60 "$RUN_LOG" >&2
  fail "missing probe_tuple_arr_capacity output line for fn=$TARGET_FN"
fi
MASK="${MASK_LINE##*mask=}"
# Both tuple slots are [f64; 2] arrays -> mask should be 3 (bits 0 and 1).
# 0 is exactly what the old 4096-entry cap would have silently left behind
# for the 4097th collected name.
if [[ "$MASK" != "3" ]]; then
  tail -n 60 "$RUN_LOG" >&2
  fail "fn=$TARGET_FN (entry $LAST of $N, past the old 4096-entry cap) has mask=$MASK, expected 3 -- its tuple-array metadata was dropped"
fi

echo "[madaros-tuple-arr-capacity-boundary] PASS: entry $LAST of $N distinct tuple-array-returning functions has mask=$MASK (correctly classified past the old 4096-entry cap)"
