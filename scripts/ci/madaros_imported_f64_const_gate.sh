#!/usr/bin/env bash
# Gate: imported-module f64 constants survive multi-module native lower (Defect A).
# docs/audit/MADAROS_NATIVE_V2_F64_REMAINING_BUGS_2026-07-20.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SOUC="${SOUC:-$ROOT/bin/souc}"
MIN="$ROOT/tests/run-pass/imported_f64_global_const.sio"
SCI="$ROOT/tests/run-pass/imported_f64_lognormal_science.sio"

if [[ ! -x "$SOUC" ]]; then
  echo "FAIL: souc not executable at $SOUC" >&2
  exit 2
fi
if [[ ! -f "$MIN" || ! -f "$SCI" ]]; then
  echo "FAIL: missing witness files" >&2
  exit 2
fi

echo "== madaros_imported_f64_const_gate: minimal multi-mod =="
out_min="$("$SOUC" run "$MIN" 2>&1)" || {
  echo "$out_min"
  echo "FAIL: minimal witness compile/run non-zero" >&2
  exit 1
}
echo "$out_min"
if ! grep -q 'IMPORTED_F64_GLOBAL_CONST_OK' <<<"$out_min"; then
  echo "FAIL: missing IMPORTED_F64_GLOBAL_CONST_OK" >&2
  exit 1
fi
if grep -q 'FAIL ' <<<"$out_min"; then
  echo "FAIL: assertion marker in minimal output" >&2
  exit 1
fi

echo "== madaros_imported_f64_const_gate: lognormal science vertical =="
out_sci="$("$SOUC" run "$SCI" 2>&1)" || {
  echo "$out_sci"
  echo "FAIL: lognormal science compile/run non-zero" >&2
  exit 1
}
echo "$out_sci"
if ! grep -q 'IMPORTED_F64_LOGNORMAL_SCIENCE_OK' <<<"$out_sci"; then
  echo "FAIL: missing IMPORTED_F64_LOGNORMAL_SCIENCE_OK" >&2
  exit 1
fi
if grep -q 'FAIL ' <<<"$out_sci"; then
  echo "FAIL: assertion marker in science output" >&2
  exit 1
fi

echo "MADAROS_IMPORTED_F64_CONST_GATE_OK"
