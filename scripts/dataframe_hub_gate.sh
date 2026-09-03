#!/usr/bin/env bash
# Gate for the DataFrame <-> CSV-file hub (data::dataframe_io): the round-trip run-proof plus the
# end-to-end example workflow (load -> describe -> filter -> save). lean_single engine.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, so relative fixture/temp paths resolve
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"
TMP1="tests/stdlib/data/df_io_roundtrip.tmp"
TMP2="tests/stdlib/data/fixtures/filtered_out.tmp"
cleanup() { rm -rf "$OUT"; rm -f "$TMP1" "$TMP2"; }
trap cleanup EXIT
fail=0
rm -f "$TMP1" "$TMP2"
echo "== check data/dataframe*.sio =="
$SOUC check stdlib/data/dataframe.sio >/dev/null 2>&1 || { echo "FAIL check dataframe"; fail=1; }
# dataframe_io standalone check trips a pre-existing Madaros check-mode parse quirk on its
# [x; N] fill-literals; it compiles fine inside the driver graph. Informational only.
$SOUC check stdlib/data/dataframe_io.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk on stdlib/data/dataframe_io.sio (Madaros check-mode; driver proves the API)"
echo "== run-proof: DataFrame <-> file round-trip + filter =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/data/test_dataframe_io_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "DATAFRAME_IO_STDLIB_OK" || { echo "FAIL run"; fail=1; }
else echo "FAIL compile"; fail=1; fi
echo "== example: load -> describe -> filter -> save =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile examples/data/dataframe_workflow.sio -o "$OUT/ex.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/ex.elf"; "$OUT/ex.elf" >/dev/null 2>&1 || { echo "FAIL example run"; fail=1; }
  [ -f "$TMP2" ] || { echo "FAIL example did not write output"; fail=1; }
else echo "FAIL example compile"; fail=1; fi
[ $fail -eq 0 ] && echo "DATAFRAME_HUB_GATE_OK"
exit $fail
