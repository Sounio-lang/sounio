#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
# NOTE: build under lean_single. stats/validation.sio uses `.len()` and other constructs the default
# Madaros engine rejects, and importing multi-module programs hit Madaros visibility-preflight; lean_single
# compiles both the module and the driver. lean_single output needs chmod +x.
# (see docs/audit/MADAROS_*_2026-07-14.md)
echo "== run-proof (lean_single, also compiles validation.sio) =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/stats/test_validation_runproof.sio -o "$OUT/sv.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/sv.elf"
  "$OUT/sv.elf" | grep -q "STATS_VALIDATION_OK" || { echo "FAIL: run-proof assertions"; fail=1; }
else echo "FAIL: run-proof compile"; fail=1; fi
[ $fail -eq 0 ] && echo "STATS_VALIDATION_GATE_OK"
exit $fail
