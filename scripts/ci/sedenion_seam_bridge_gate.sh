#!/usr/bin/env bash
# ADR-008: claim = BRIDGE OK on Sounio; Python six-way/incidence corroboration soft unless HARD=1.
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_seam_bridge.sio | grep -E '^(EQUIV_OK|N_|BRIDGE)' | sort > "$WORK/souc.txt" || true
fail=0
grep -q '^BRIDGE OK' "$WORK/souc.txt" || { echo "CLAIM FAIL: verdict != BRIDGE OK"; fail=1; }
if python3 scripts/research/sedenion_seam_bridge_oracle.py > "$WORK/py_all.txt" 2>/dev/null; then
  grep -E '^(EQUIV_OK|N_|BRIDGE)' "$WORK/py_all.txt" | sort > "$WORK/py.txt" || true
  sounio_foreign_diff "$WORK/souc.txt" "$WORK/py.txt" "seam bridge" || fail=1
  [ "$(grep '^SIXWAY_OK ' "$WORK/py_all.txt" | awk '{print $2}')" = "1" ] \
    || sounio_foreign_mismatch "oracle SIXWAY_OK != 1" || fail=1
  [ "$(grep '^INCIDENCE_OK ' "$WORK/py_all.txt" | awk '{print $2}')" = "1" ] \
    || sounio_foreign_mismatch "oracle INCIDENCE_OK != 1" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "seam bridge gate: FAIL"; exit 1; }
echo "seam bridge gate: PASS"
