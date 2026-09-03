#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== gum_report negative =="
if $SOUC compile tests/stdlib/epistemic/test_gum_negative.sio -o "$OUT/g.elf"; then
  "$OUT/g.elf" | grep -q -- "delta = -5.000000" || { echo "FAIL: gum_report negative"; fail=1; }
else echo "FAIL: gum negative compile"; fail=1; fi
echo "== quantity_show negative =="
if $SOUC compile tests/stdlib/units/test_units_negative.sio -o "$OUT/u.elf"; then
  "$OUT/u.elf" | grep -q -- "offset = -10.000000" || { echo "FAIL: quantity_show negative"; fail=1; }
else echo "FAIL: units negative compile"; fail=1; fi
[ $fail -eq 0 ] && echo "NEGATIVE_DISPLAY_GATE_OK"
exit $fail
