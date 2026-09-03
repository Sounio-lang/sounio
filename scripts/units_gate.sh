#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check units/lib.sio =="
$SOUC check stdlib/units/lib.sio || fail=1
echo "== run-proof driver =="
if $SOUC compile tests/stdlib/units/test_units_stdlib.sio -o "$OUT/tu.elf"; then
  "$OUT/tu.elf" | grep -q "UNITS_STDLIB_OK" || { echo "FAIL: driver assertions"; fail=1; }
else echo "FAIL: driver compile"; fail=1; fi
echo "== consumer example =="
if $SOUC compile examples/units/dimensional_report.sio -o "$OUT/dr.elf"; then
  "$OUT/dr.elf" >/dev/null || { echo "FAIL: example run"; fail=1; }
else echo "FAIL: example compile"; fail=1; fi
[ $fail -eq 0 ] && echo "UNITS_GATE_OK"
exit $fail
