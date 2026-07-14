#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check distributions.sio =="
$SOUC check stdlib/prob/distributions.sio || fail=1
# NOTE: native compile needs lean_single (Madaros native scale limit — see
# docs/audit/MADAROS_NATIVE_MULTIMODULE_SCALE_2026-07-14.md); lean_single output needs chmod +x.
echo "== run-proof driver (lean_single) =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/prob/test_prob_stdlib.sio -o "$OUT/tp.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/tp.elf"
  "$OUT/tp.elf" | grep -q "PROB_STDLIB_OK" || { echo "FAIL: driver assertions"; fail=1; }
else echo "FAIL: driver compile"; fail=1; fi
echo "== consumer example (lean_single) =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile examples/prob/distribution_report.sio -o "$OUT/dr.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/dr.elf"
  "$OUT/dr.elf" >/dev/null || { echo "FAIL: example run"; fail=1; }
else echo "FAIL: example compile"; fail=1; fi
[ $fail -eq 0 ] && echo "PROB_GATE_OK"
exit $fail
