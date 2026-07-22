#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
engine_info="$($SOUC info 2>&1)"
if ! grep -qF 'Madaros v' <<<"$engine_info"; then
  echo "FAIL: probability gate requires default Madaros" >&2
  printf '%s\n' "$engine_info" >&2
  exit 1
fi
echo "== check distributions.sio =="
$SOUC check stdlib/prob/distributions.sio || fail=1
echo "== run-proof driver (Madaros) =="
if $SOUC compile tests/stdlib/prob/test_prob_stdlib.sio -o "$OUT/tp.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/tp.elf"
  "$OUT/tp.elf" | grep -q "PROB_STDLIB_OK" || { echo "FAIL: driver assertions"; fail=1; }
else echo "FAIL: driver compile"; fail=1; fi
echo "== consumer example (Madaros) =="
if $SOUC compile examples/prob/distribution_report.sio -o "$OUT/dr.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/dr.elf"
  "$OUT/dr.elf" >/dev/null || { echo "FAIL: example run"; fail=1; }
else echo "FAIL: example compile"; fail=1; fi
[ $fail -eq 0 ] && echo "PROB_GATE_OK"
exit $fail
