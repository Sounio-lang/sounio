#!/usr/bin/env bash
# ADR-008: claim = SEDEXT OK on Sounio; Python diff corroboration soft unless HARD=1.
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/l.elf" >/dev/null 2>&1; chmod +x "$WORK/l.elf"; "$WORK/l.elf" 2>/dev/null; fi }
run_souc tests/run-pass/sedenion_ladder_extension.sio | grep -E '^(B1_OK|OCT_RANK|SED_RANK|SEDEXT)' | sort > "$WORK/souc.txt"
fail=0
grep -q '^SEDEXT OK' "$WORK/souc.txt" || { echo "CLAIM FAIL: missing SEDEXT OK"; fail=1; }
if python3 scripts/research/sedenion_ladder_extension_oracle.py 2>/dev/null | grep -E '^(B1_OK|OCT_RANK|SED_RANK|SEDEXT)' | sort > "$WORK/py.txt"; then
  sounio_foreign_diff "$WORK/souc.txt" "$WORK/py.txt" "ladder" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "sedenion ladder extension gate: FAIL"; exit 1; }
echo "sedenion ladder extension gate: PASS"
