#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check erf.sio =="; $SOUC check stdlib/special/erf.sio || fail=1
echo "== run-proof =="
if $SOUC compile tests/stdlib/special/test_erf_stdlib.sio -o "$OUT/te.elf"; then
  "$OUT/te.elf" | grep -q "ERF_STDLIB_OK" || { echo "FAIL: assertions"; fail=1; }
else echo "FAIL: compile"; fail=1; fi
echo "== example =="
if $SOUC compile examples/special/erf_report.sio -o "$OUT/er.elf"; then "$OUT/er.elf" >/dev/null || { echo "FAIL: example"; fail=1; }
else echo "FAIL: example compile"; fail=1; fi
[ $fail -eq 0 ] && echo "SPECIAL_ERF_GATE_OK"
exit $fail
