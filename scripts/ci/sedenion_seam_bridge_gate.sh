#!/usr/bin/env bash
# Cross-toolchain gate: the e8-seam bridge (Frente B). souc (sparse core) vs oracle (full six-way + incidence).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_seam_bridge.sio | grep -E '^(EQUIV_OK|N_|BRIDGE)' | sort > "$WORK/souc.txt"
python3 scripts/research/sedenion_seam_bridge_oracle.py > "$WORK/py_all.txt"
grep -E '^(EQUIV_OK|N_|BRIDGE)' "$WORK/py_all.txt" | sort > "$WORK/py.txt"
fail=0
diff -q "$WORK/souc.txt" "$WORK/py.txt" >/dev/null || { echo "MISMATCH:"; diff "$WORK/souc.txt" "$WORK/py.txt"; fail=1; }
grep -q '^BRIDGE OK' "$WORK/souc.txt" || { echo "verdict != BRIDGE OK"; fail=1; }
[ "$(grep '^SIXWAY_OK ' "$WORK/py_all.txt" | awk '{print $2}')" = "1" ] || { echo "full six-way equivalence failed"; fail=1; }
[ "$(grep '^INCIDENCE_OK ' "$WORK/py_all.txt" | awk '{print $2}')" = "1" ] || { echo "4-regular incidence failed"; fail=1; }
[ "$fail" -eq 0 ] || { echo "seam bridge gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: operator non-anticommutation = state zero-division = off the e8 seam (42=42=42);"
echo "  full six-way equivalence (incl. det/spec) + 4-regular self-paired quartet incidence (168)."
echo "seam bridge gate: PASS"
