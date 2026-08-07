#!/usr/bin/env bash
# ADR-008: claim = CL8 OK on Sounio; Python/DIM256 corroboration soft unless HARD=1.
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_clifford8.sio | grep -E '^(CL8_|NONANTI|GENS|RANK|Q1_|CL8 )' | sort > "$WORK/souc.txt" || true
fail=0
grep -q '^CL8 OK' "$WORK/souc.txt" || { echo "CLAIM FAIL: souc verdict != CL8 OK"; fail=1; }
if python3 scripts/research/sedenion_clifford8_oracle.py > "$WORK/py_all.txt" 2>/dev/null; then
  grep -E '^(CL8_|NONANTI|GENS|RANK|Q1_|CL8 )' "$WORK/py_all.txt" | sort > "$WORK/py.txt" || true
  sounio_foreign_diff "$WORK/souc.txt" "$WORK/py.txt" "clifford8" || fail=1
  dim=$(grep '^DIM256 ' "$WORK/py_all.txt" | awk '{print $2}')
  [ "$dim" = "256" ] || sounio_foreign_mismatch "oracle DIM256 != 256 (got $dim)" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "clifford8 gate: FAIL"; exit 1; }
echo "clifford8 gate: PASS"
