#!/usr/bin/env bash
# Copilot review (PR #2516), finding D on the fourth review round of
# fix/madaros-f64-array-element-kind: lower_program_bodies_filtered_ref (the
# streaming single-function lowering lane -- reached only via the NON-flat
# lower_program_function_body_from_summary_with_epistemic_ref, not via
# `souc run` or even --probe-native-streaming, which both go through the
# FLAT helper's different lane) had no direct, scriptable coverage. This
# gate drives it through the --probe-filtered-body-lower CLI flag added in
# main.sio for exactly this purpose.
#
# Round 6, finding 4 (Copilot again): the original version of this gate
# asserted `float_reg_marks > 0` for fn=main as proof the collector fired.
# That is not airtight -- the fixture's own ordinary f64 literals (2.0, 0.5,
# ...) each call emit_f64_const, which ALSO emits ir_mark_float_reg, so the
# assertion stayed positive even with the collector call removed entirely
# (confirmed below by actually removing it and re-running this gate before
# trusting the fix). This now asserts `tuple_arr_mask=3` for fn=pair
# instead: lower_fn_tuple_f64_array_mask(pair) is a direct query of
# LOWER_FN_TUPLE_ARR_MASK, populated ONLY by
# lower_fn_tuple_f64_arrays_collect (called at the top of
# lower_program_bodies_filtered_ref) -- not by any literal in the source --
# so a nonzero mask here is specific evidence the collector ran through
# this exact lane, not a body-wide instruction count that any float
# arithmetic would also satisfy.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_FILTERED_BODY_GATE_KEEP:-0}"
SOURCE="$ROOT_DIR/tests/run-pass/madaros_tuple_f64_array_destructure.sio"

fail() {
  echo "[madaros-tuple-arr-filtered-body-lower] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_FILTERED_BODY_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_FILTERED_BODY_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-filtered-body.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_FILTERED_BODY_GATE_BIN:-$WORK/madaros}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_FILTERED_BODY_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "Madaros is missing or not executable: $MADAROS_ELF"

set +e
"$MADAROS_ELF" --probe-filtered-body-lower "$SOURCE" >"$WORK/probe.log" 2>&1
probe_rc=$?
set -e

if [[ "$probe_rc" != "0" ]]; then
  cat "$WORK/probe.log" >&2
  fail "--probe-filtered-body-lower exited rc=$probe_rc"
fi

MASK_LINE="$(grep -E '^probe_filtered_body_lower: body_ok fn=pair .*tuple_arr_mask=[0-9]+$' "$WORK/probe.log" || true)"
if [[ -z "$MASK_LINE" ]]; then
  cat "$WORK/probe.log" >&2
  fail "missing body_ok line with tuple_arr_mask for fn=pair"
fi
MASK="${MASK_LINE##*tuple_arr_mask=}"
# `pair` returns ([f64;2], [f64;2]) -- both tuple slots are f64 arrays, so
# the correct mask is 3 (bits 0 and 1). 0 is exactly what an unwired
# collector (the round-3 regression this gate guards) would leave behind.
if [[ "$MASK" != "3" ]]; then
  cat "$WORK/probe.log" >&2
  fail "fn=pair has tuple_arr_mask=$MASK, expected 3 -- lower_program_bodies_filtered_ref did not populate the tuple-array table for its own item list"
fi

echo "[madaros-tuple-arr-filtered-body-lower] PASS: lower_program_bodies_filtered_ref populated tuple_arr_mask=$MASK for fn=pair"
