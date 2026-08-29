#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check matnm.sio =="
$SOUC check stdlib/linalg/matnm.sio || fail=1
echo "== run-proof driver =="
if $SOUC compile tests/stdlib/linalg/test_matnm_stdlib.sio -o "$OUT/tm.elf"; then
  "$OUT/tm.elf" | grep -q "MATNM_STDLIB_OK" || { echo "FAIL: driver assertions"; fail=1; }
else echo "FAIL: driver compile"; fail=1; fi
echo "== consumer example =="
if $SOUC compile examples/linalg/solve_report.sio -o "$OUT/sr.elf"; then
  "$OUT/sr.elf" >/dev/null || { echo "FAIL: example run"; fail=1; }
else echo "FAIL: example compile"; fail=1; fi
[ $fail -eq 0 ] && echo "LINALG_GATE_OK"
exit $fail
