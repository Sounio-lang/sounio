#!/usr/bin/env bash
# ADR-008: claim = FAMILYS3 OK on Sounio; Python soft unless HARD=1.
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/x.elf" >/dev/null 2>&1; chmod +x "$WORK/x.elf"; "$WORK/x.elf" 2>/dev/null; fi }
run_souc tests/run-pass/gresnigt_family_s3.sio | grep -E '^(PSI_|FAMILYS3)' | sort > "$WORK/souc.txt" || true
fail=0
grep -q '^FAMILYS3 OK' "$WORK/souc.txt" || { echo "CLAIM FAIL: missing FAMILYS3 OK"; fail=1; }
if python3 scripts/research/gresnigt_family_s3_oracle.py > "$WORK/py_all.txt" 2>/dev/null; then
  grep -E '^(PSI_|FAMILYS3)' "$WORK/py_all.txt" | sort > "$WORK/py.txt" || true
  sounio_foreign_diff "$WORK/souc.txt" "$WORK/py.txt" "gresnigt family s3" || fail=1
  [ "$(grep '^COMM_PHI_ZERO ' "$WORK/py_all.txt" | awk '{print $2}')" = "1" ] \
    || sounio_foreign_mismatch "oracle COMM_PHI_ZERO != 1" || fail=1
  [ "$(grep '^COMM_PSI_NONZERO ' "$WORK/py_all.txt" | awk '{print $2}')" = "1" ] \
    || sounio_foreign_mismatch "oracle COMM_PSI_NONZERO != 1" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "gresnigt family s3 gate: FAIL"; exit 1; }
echo "gresnigt family s3 gate: PASS"
