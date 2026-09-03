#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check filter.sio =="; $SOUC check stdlib/signal/filter.sio || fail=1
echo "== run-proof =="
if $SOUC compile tests/stdlib/signal/test_filter_stdlib.sio -o "$OUT/tf.elf"; then
  "$OUT/tf.elf" | grep -q "FILTER_STDLIB_OK" || { echo "FAIL: assertions"; fail=1; }
else echo "FAIL: compile"; fail=1; fi
echo "== example =="
if $SOUC compile examples/signal/filter_report.sio -o "$OUT/fr.elf"; then "$OUT/fr.elf" >/dev/null || { echo "FAIL: example"; fail=1; }
else echo "FAIL: example compile"; fail=1; fi
[ $fail -eq 0 ] && echo "SIGNAL_FILTER_GATE_OK"
exit $fail
