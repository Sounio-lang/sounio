#!/usr/bin/env bash
# Gate for the capstone measurement-QC workflow example. lean_single engine.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== run: clean -> combine -> decide =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile examples/data/measurement_qc_workflow.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | tee "$OUT/run.txt"; grep -q "MEASUREMENT_QC_WORKFLOW_OK" "$OUT/run.txt" || { echo "FAIL run"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "MEASUREMENT_QC_WORKFLOW_GATE_OK"
exit $fail
