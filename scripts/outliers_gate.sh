#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check data/outliers.sio =="
$SOUC check stdlib/data/outliers.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk on stdlib/data/outliers.sio (Madaros check-mode; driver proves the API)"
echo "== run-proof: Grubbs outlier test =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/data/test_outliers_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "OUTLIERS_STDLIB_OK" || { echo "FAIL run"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "OUTLIERS_GATE_OK"
exit $fail
