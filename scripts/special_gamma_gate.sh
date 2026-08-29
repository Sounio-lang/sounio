#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check gamma.sio =="; $SOUC check stdlib/special/gamma.sio || fail=1
echo "== run-proof =="
if $SOUC compile tests/stdlib/special/test_gamma_stdlib.sio -o "$OUT/tg.elf"; then
  "$OUT/tg.elf" | grep -q "GAMMA_STDLIB_OK" || { echo "FAIL: assertions"; fail=1; }
else echo "FAIL: compile"; fail=1; fi
echo "== example =="
if $SOUC compile examples/special/gamma_report.sio -o "$OUT/gr.elf"; then "$OUT/gr.elf" >/dev/null || { echo "FAIL: example"; fail=1; }
else echo "FAIL: example compile"; fail=1; fi
[ $fail -eq 0 ] && echo "SPECIAL_GAMMA_GATE_OK"
exit $fail
