#!/usr/bin/env bash
# KL-2 / #2389 — IEEE 754 f64 special values: PF-aware compares and print_f64 inf/nan.
#
# Witnesses:
#   tests/run-pass/ieee_f64_nan_compare.sio
#   tests/run-pass/ieee_f64_print_inf.sio
#   tests/run-pass/ieee_f64_print_nan.sio
#
# Runs on default Madaros and on lean_single (SOUNIO_SOUC_ENGINE=lean_single).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"

NAN_CMP="$ROOT/tests/run-pass/ieee_f64_nan_compare.sio"
PRINT_INF="$ROOT/tests/run-pass/ieee_f64_print_inf.sio"
PRINT_NAN="$ROOT/tests/run-pass/ieee_f64_print_nan.sio"

for f in "$NAN_CMP" "$PRINT_INF" "$PRINT_NAN"; do
  [[ -f "$f" ]] || { echo "FAIL: missing witness $f" >&2; exit 2; }
done

run_engine() {
  local label="$1"
  shift
  echo "== engine: $label =="
  local out rc
  out="$("$@" run "$NAN_CMP" 2>&1)" || {
    echo "$out"
    echo "FAIL: $label nan_compare" >&2
    exit 1
  }
  echo "$out"
  grep -q 'IEEE_F64_NAN_CMP_OK' <<<"$out" || {
    echo "FAIL: $label missing IEEE_F64_NAN_CMP_OK" >&2
    exit 1
  }

  out="$("$@" run "$PRINT_INF" 2>&1)" || {
    echo "$out"
    echo "FAIL: $label print_inf" >&2
    exit 1
  }
  echo "$out"
  grep -q '^START$' <<<"$out" || { echo "FAIL: $label print_inf START" >&2; exit 1; }
  grep -q '^inf$' <<<"$out" || { echo "FAIL: $label print_inf body (want inf)" >&2; exit 1; }
  grep -q '^END$' <<<"$out" || { echo "FAIL: $label print_inf END" >&2; exit 1; }

  out="$("$@" run "$PRINT_NAN" 2>&1)" || {
    echo "$out"
    echo "FAIL: $label print_nan" >&2
    exit 1
  }
  echo "$out"
  grep -q '^START$' <<<"$out" || { echo "FAIL: $label print_nan START" >&2; exit 1; }
  grep -q '^nan$' <<<"$out" || { echo "FAIL: $label print_nan body (want nan)" >&2; exit 1; }
  grep -q '^END$' <<<"$out" || { echo "FAIL: $label print_nan END" >&2; exit 1; }
}

echo "== madaros_ieee_f64_special_values_gate =="
echo "madaros: $($SOUC --version 2>&1 | head -1)"
run_engine madaros "$SOUC"
run_engine lean_single env SOUNIO_SOUC_ENGINE=lean_single "$SOUC"

echo "PASS madaros_ieee_f64_special_values_gate"
echo "MADAROS_IEEE_F64_SPECIAL_VALUES_GATE_OK"
