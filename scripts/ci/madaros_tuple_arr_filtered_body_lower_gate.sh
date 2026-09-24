#!/usr/bin/env bash
# Copilot review (PR #2516), finding D on the fourth review round of
# fix/madaros-f64-array-element-kind: lower_program_bodies_filtered_ref (the
# streaming single-function lowering lane -- reached only via the NON-flat
# lower_program_function_body_from_summary_with_epistemic_ref, not via
# `souc run` or even --probe-native-streaming, which both go through the
# FLAT helper's different lane) had no direct, scriptable coverage. This
# gate drives it through the --probe-filtered-body-lower CLI flag added in
# main.sio for exactly this purpose and asserts that the destructured f64
# array actually got classified as float THROUGH THAT LANE (a nonzero
# float_reg_marks count for `main`), not just that lowering didn't error.

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

grep -Eq '^probe_filtered_body_lower: body_ok fn=main instrs=[0-9]+ float_reg_marks=[0-9]+$' "$WORK/probe.log" || {
  cat "$WORK/probe.log" >&2
  fail "missing body_ok line for fn=main"
}

marks="$(grep -E '^probe_filtered_body_lower: body_ok fn=main ' "$WORK/probe.log" | grep -oE 'float_reg_marks=[0-9]+' | cut -d= -f2)"
[[ -n "$marks" && "$marks" -gt 0 ]] || {
  cat "$WORK/probe.log" >&2
  fail "float_reg_marks=$marks for fn=main -- lower_program_bodies_filtered_ref did not mark the destructured f64 array as float"
}

echo "[madaros-tuple-arr-filtered-body-lower] PASS: lower_program_bodies_filtered_ref marked $marks float register(s) for fn=main"
