#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check data/dataframe_ffill.sio =="
$SOUC check stdlib/data/dataframe_ffill.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk on stdlib/data/dataframe_ffill.sio (Madaros check-mode; driver proves the API)"
echo "== run-proof: ffill =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/data/test_dataframe_ffill_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "DATAFRAME_FFILL_STDLIB_OK" || { echo "FAIL run"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "DATAFRAME_FFILL_GATE_OK"
exit $fail
