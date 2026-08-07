#!/usr/bin/env bash
# ADR-008: claim = Sounio SPECTRA OK; Python diff corroboration soft unless HARD=1.
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
run_souc tests/run-pass/sedenion_spectra.sio | grep -E '^(FIBM|K7M|VERTS|SPECTRA)' | sort > "$WORK/souc.txt"
fail=0
grep -q '^SPECTRA OK' "$WORK/souc.txt" || { echo "CLAIM FAIL: souc missing SPECTRA OK"; fail=1; }
if python3 scripts/research/sedenion_spectra_oracle.py 2>/dev/null | grep -E '^(FIBM|K7M|VERTS|SPECTRA)' | sort > "$WORK/py.txt"; then
  sounio_foreign_diff "$WORK/souc.txt" "$WORK/py.txt" "spectra" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "spectra gate: FAIL"; exit 1; }
echo "spectra gate: PASS"
