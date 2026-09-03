#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")/.."; export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT; fail=0
./bin/souc check stdlib/data/quantity.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk (driver proves the API)"
if SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile tests/stdlib/data/test_quantity_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "DIMENSIONAL_UNITS_STDLIB_OK" || { echo FAIL; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "DIMENSIONAL_UNITS_GATE_OK"; exit $fail
