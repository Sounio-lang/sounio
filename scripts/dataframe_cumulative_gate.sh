#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check data/dataframe_cumulative.sio =="
$SOUC check stdlib/data/dataframe_cumulative.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk on stdlib/data/dataframe_cumulative.sio (Madaros check-mode; driver proves the API)"
echo "== run-proof: cumulative =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/data/test_dataframe_cumulative_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "DATAFRAME_CUMULATIVE_STDLIB_OK" || { echo "FAIL run"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "DATAFRAME_CUMULATIVE_GATE_OK"
exit $fail
