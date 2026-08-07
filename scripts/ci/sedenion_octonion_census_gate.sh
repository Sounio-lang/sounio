#!/usr/bin/env bash
# ADR-008: claim = OCTCENSUS OK on Sounio; Python diff corroboration soft unless HARD=1.
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/o.elf" >/dev/null 2>&1; chmod +x "$WORK/o.elf"; "$WORK/o.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_octonion_census.sio | grep -E '^(NSUB|ZDFREE|QUASI|PURE|QUAT|OCTCENSUS)' | sort > "$WORK/souc.txt" || true
fail=0
grep -q '^OCTCENSUS OK' "$WORK/souc.txt" || { echo "CLAIM FAIL: missing OCTCENSUS OK"; fail=1; }
if python3 scripts/research/sedenion_octonion_census_oracle.py 2>/dev/null | grep -E '^(NSUB|ZDFREE|QUASI|PURE|QUAT|OCTCENSUS)' | sort > "$WORK/py.txt"; then
  sounio_foreign_diff "$WORK/souc.txt" "$WORK/py.txt" "octonion census" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "octonion census gate: FAIL"; exit 1; }
echo "octonion census gate: PASS"
