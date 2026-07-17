#!/usr/bin/env bash
# Gate for stdlib data::dataframe_sort descending / mixed-direction sort. lean_single engine.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check data/dataframe_sort.sio =="
$SOUC check stdlib/data/dataframe_sort.sio >/dev/null 2>&1 || { echo "FAIL check"; fail=1; }
echo "== run-proof: descending + mixed-direction sort =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/data/test_dataframe_sort_desc_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "DATAFRAME_SORT_DESC_STDLIB_OK" || { echo "FAIL run"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "DATAFRAME_SORT_DESC_GATE_OK"
exit $fail
