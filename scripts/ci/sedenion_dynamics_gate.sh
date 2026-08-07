#!/usr/bin/env bash
# ADR-008: claim clock = Sounio DYNAMICS OK (+ emitted lines non-empty);
# Python/diff is corroboration (hard-fail only if SOUNIO_FOREIGN_ORACLE_HARD=1).
# Substrate dynamics of ZD-geometry graphs (spanning-tree tau + walk counts).
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/ci/lib_sounio_claim_oracle.sh
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"

cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/s.elf" >/dev/null 2>&1; chmod +x "$WORK/s.elf"; "$WORK/s.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_dynamics.sio | grep -E '^(SPANTREE|FIB_|K7_|VERTS|DYNAMICS)' | sort > "$WORK/souc.txt"
fail=0
grep -q '^DYNAMICS OK' "$WORK/souc.txt" || { echo "CLAIM FAIL: souc missing DYNAMICS OK"; fail=1; }
[ "$(wc -l < "$WORK/souc.txt")" -ge 1 ] || { echo "CLAIM FAIL: empty souc dynamics emit"; fail=1; }

if python3 -c 'import sys' 2>/dev/null; then
  python3 scripts/research/sedenion_dynamics_oracle.py | grep -E '^(SPANTREE|FIB_|K7_|VERTS|DYNAMICS)' | sort > "$WORK/py.txt" || true
  if [ -s "$WORK/py.txt" ]; then
    sounio_foreign_diff "$WORK/souc.txt" "$WORK/py.txt" "dynamics" || fail=1
  fi
fi

[ "$fail" -eq 0 ] || { echo "dynamics gate: FAIL"; exit 1; }
echo "dynamics gate: PASS (Sounio claim OK; foreign corroboration soft unless HARD=1)"
