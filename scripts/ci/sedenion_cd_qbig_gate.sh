#!/usr/bin/env bash
# ADR-008: claim = CDQBIG OK on Sounio; Python residue match is corroboration.
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
OUT="$(run_souc tests/run-pass/sedenion_cd_qbig.sio)"
echo "$OUT" | grep -aE '^(RES|ANNIHILATION_C1) ' > "$WORK/souc.txt" || true
fail=0
echo "$OUT" | grep -qa '^CDQBIG OK' || { echo "CLAIM FAIL: verdict != CDQBIG OK"; fail=1; }
[ -s "$WORK/souc.txt" ] || { echo "CLAIM FAIL: empty RES emit"; fail=1; }
if python3 scripts/research/sedenion_cd_qbig_oracle.py 2>/dev/null | grep -E '^(RES|ANNIHILATION_C1) ' > "$WORK/py.txt"; then
  sounio_foreign_diff "$WORK/souc.txt" "$WORK/py.txt" "cd-qbig" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "cd-qbig gate: FAIL"; exit 1; }
echo "cd-qbig gate: PASS"
