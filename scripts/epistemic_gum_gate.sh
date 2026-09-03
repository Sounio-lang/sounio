#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

# NOTE: gum.sio's own embedded self-test segfaults at runtime on Madaros v0.80.0
# (pre-existing; see docs/audit/MADAROS_MULTIMODULE_PRINT_IMPORT_BUGS_2026-07-13.md, Defect 4).
# We therefore verify the module via the importing run-proof driver below, not by running gum.sio.
echo "== check gum.sio =="
$SOUC check stdlib/epistemic/gum.sio || fail=1

echo "== run-proof driver (combine + report) =="
if $SOUC compile tests/stdlib/epistemic/test_gum_stdlib.sio -o "$OUT/tg.elf"; then
  "$OUT/tg.elf" | grep -q "GUM_STDLIB_OK" || { echo "FAIL: driver assertions"; fail=1; }
else echo "FAIL: driver compile"; fail=1; fi

echo "== run-proof ops (add/sub/mul/div/scale, triangular, sensitivity, u99) =="
if $SOUC compile tests/stdlib/epistemic/test_gum_ops.sio -o "$OUT/ops.elf"; then
  "$OUT/ops.elf" | grep -q "GUM_OPS_OK" || { echo "FAIL: ops assertions"; fail=1; }
else echo "FAIL: ops compile"; fail=1; fi

echo "== consumer example =="
if $SOUC compile examples/epistemic/gum_measurement_chain.sio -o "$OUT/chain.elf"; then
  "$OUT/chain.elf" >/dev/null || { echo "FAIL: example run"; fail=1; }
else echo "FAIL: example compile"; fail=1; fi

echo "== regression: vancomycin (shares epistemic package) =="
$SOUC check stdlib/clinical/vancomycin_pbpk.sio || { echo "REGRESSION"; fail=1; }

[ $fail -eq 0 ] && echo "EPISTEMIC_GUM_GATE_OK"
exit $fail
