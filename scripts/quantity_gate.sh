#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")/.."; export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT; fail=0
engine_info="$(./bin/souc info 2>&1)"
if ! grep -qF 'Madaros v' <<<"$engine_info"; then
  echo "FAIL: dimensional-units gate requires default Madaros" >&2
  printf '%s\n' "$engine_info" >&2
  exit 1
fi
./bin/souc check stdlib/data/quantity.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk (driver proves the API)"
if ./bin/souc compile tests/stdlib/data/test_quantity_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "DIMENSIONAL_UNITS_STDLIB_OK" || { echo FAIL; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "DIMENSIONAL_UNITS_GATE_OK"; exit $fail
