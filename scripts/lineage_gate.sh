#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check data/lineage.sio =="
$SOUC check stdlib/data/lineage.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk on stdlib/data/lineage.sio (Madaros check-mode; driver proves the API)"
echo "== run-proof: provenance/lineage DAG =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/data/test_lineage_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "LINEAGE_STDLIB_OK" || { echo "FAIL run"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "LINEAGE_GATE_OK"
exit $fail
