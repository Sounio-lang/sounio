#!/usr/bin/env bash
# Gate: imported-module f64 constants survive multi-module Madaros.
# Defect A (Wave11e suppress GLOBAL_VAR_INIT) + Defect A′ (Wave11 multi-mod BSS remap).
# docs/audit/MADAROS_NATIVE_V2_F64_REMAINING_BUGS_2026-07-20.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SOUC="${SOUC:-$ROOT/bin/souc}"
MIN="$ROOT/tests/run-pass/imported_f64_global_const.sio"
SCI="$ROOT/tests/run-pass/imported_f64_lognormal_science.sio"
BSS="$ROOT/tests/run-pass/imported_module_f64_const.sio"
BARE="$ROOT/tests/run-pass/imported_module_f64_const_bare_ident.sio"

if [[ ! -x "$SOUC" ]]; then
  echo "FAIL: souc not executable at $SOUC" >&2
  exit 2
fi
if [[ ! -f "$MIN" || ! -f "$SCI" || ! -f "$BSS" || ! -f "$BARE" ]]; then
  echo "FAIL: missing witness files" >&2
  exit 2
fi

echo "== madaros_imported_f64_const_gate: minimal multi-mod (Defect A) =="
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

echo "== madaros_imported_f64_const_gate: multi-mod BSS remap (Defect A′) =="
out_bss="$("$SOUC" run "$BSS" 2>&1)" || {
  echo "$out_bss"
  echo "FAIL: multi-mod BSS witness compile/run non-zero" >&2
  exit 1
}
echo "$out_bss"
if ! grep -q 'IMPORTED_F64_CONST_OK' <<<"$out_bss"; then
  echo "FAIL: missing IMPORTED_F64_CONST_OK" >&2
  exit 1
fi
if grep -q 'FAIL ' <<<"$out_bss"; then
  echo "FAIL: assertion marker in BSS multi-mod output" >&2
  exit 1
fi
# Distinct A/B bits (last-init-wins collision would print B for both).
if ! grep -q '4609434218613702656' <<<"$out_bss"; then
  echo "FAIL: missing A_CONST bits 1.5" >&2
  exit 1
fi
if ! grep -q '4612811918334230528' <<<"$out_bss"; then
  echo "FAIL: missing B_CONST bits 2.5" >&2
  exit 1
fi

echo "== madaros_imported_f64_const_gate: bare cross-mod Ident (Wave13) =="
out_bare="$("$SOUC" run "$BARE" 2>&1)" || {
  echo "$out_bare"
  echo "FAIL: bare cross-mod Ident witness compile/run non-zero" >&2
  exit 1
}
echo "$out_bare"
if ! grep -q 'BARE_CROSSMOD_F64_IDENT_OK' <<<"$out_bare"; then
  echo "FAIL: missing BARE_CROSSMOD_F64_IDENT_OK" >&2
  exit 1
fi
if grep -q 'FAIL ' <<<"$out_bare"; then
  echo "FAIL: assertion marker in bare Ident output" >&2
  exit 1
fi
if ! grep -q '4609434218613702656' <<<"$out_bare"; then
  echo "FAIL: missing A_CONST bits 1.5 on bare Ident path" >&2
  exit 1
fi

echo "MADAROS_IMPORTED_F64_CONST_GATE_OK"
echo "PASS madaros_imported_f64_const_gate"
