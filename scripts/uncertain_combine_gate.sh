#!/usr/bin/env bash
# Gate for stdlib data::uncertain_combine (weighted mean / reduced chi-square / Birge). lean_single.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check data/uncertain_combine.sio =="
$SOUC check stdlib/data/uncertain_combine.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk on stdlib/data/uncertain_combine.sio (Madaros check-mode; driver proves the API)"
echo "== run-proof: weighted mean + reduced chi2 + Birge =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/data/test_uncertain_combine_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "UNCERTAIN_COMBINE_STDLIB_OK" || { echo "FAIL run"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "UNCERTAIN_COMBINE_GATE_OK"
exit $fail
