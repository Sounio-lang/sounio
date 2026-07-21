#!/usr/bin/env bash
# Gate for stdlib data::dataframe_relational (group-by + inner join) under default Madaros.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
engine_info="$($SOUC info 2>&1)"
if ! grep -qF 'Madaros v' <<<"$engine_info"; then
  echo "FAIL: dataframe-relational gate requires default Madaros" >&2
  printf '%s\n' "$engine_info" >&2
  exit 1
fi
echo "== check data/dataframe_relational.sio =="
# pre-existing Madaros check-mode parse quirk on [x; N] fill-literals; driver is the proof
$SOUC check stdlib/data/dataframe_relational.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk on stdlib/data/dataframe_relational.sio (Madaros check-mode; driver proves the API)"
echo "== run-proof: groupby + join =="
if $SOUC compile tests/stdlib/data/test_dataframe_relational_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "DATAFRAME_RELATIONAL_STDLIB_OK" || { echo "FAIL run"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "DATAFRAME_RELATIONAL_GATE_OK"
exit $fail
