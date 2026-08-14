#!/usr/bin/env bash
# ADR-008: claim = GRESNIGT OK on Sounio; Python diff corroboration soft unless HARD=1.
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/g.elf" >/dev/null 2>&1; chmod +x "$WORK/g.elf"; "$WORK/g.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_gresnigt_octonions.sio | grep -E '^(AUT_OK|ORD3|FIX_|G2_|OCTS_|CYCLE_|GRESNIGT)' | sort > "$WORK/souc.txt"
fail=0
grep -q '^GRESNIGT OK' "$WORK/souc.txt" || { echo "CLAIM FAIL: verdict != GRESNIGT OK"; fail=1; }
if python3 scripts/research/sedenion_gresnigt_octonions_oracle.py 2>/dev/null | grep -E '^(AUT_OK|ORD3|FIX_|G2_|OCTS_|CYCLE_|GRESNIGT)' | sort > "$WORK/py.txt"; then
  sounio_foreign_diff "$WORK/souc.txt" "$WORK/py.txt" "gresnigt" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "gresnigt gate: FAIL"; exit 1; }
echo "gresnigt gate: PASS"
