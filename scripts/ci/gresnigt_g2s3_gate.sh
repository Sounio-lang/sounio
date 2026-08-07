#!/usr/bin/env bash
# ADR-008: claim = G2S3 OK on Sounio; Python/diff soft unless HARD=1.
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/x.elf" >/dev/null 2>&1; chmod +x "$WORK/x.elf"; "$WORK/x.elf" 2>/dev/null; fi }
run_souc tests/run-pass/gresnigt_g2s3.sio | grep -E '^(EPS_|S3_|COMM_|PSI_|ONLY_|G2S3)' | sort > "$WORK/souc.txt" || true
fail=0
grep -q '^G2S3 OK' "$WORK/souc.txt" || { echo "CLAIM FAIL: missing G2S3 OK"; fail=1; }
if python3 scripts/research/gresnigt_g2s3_oracle.py 2>/dev/null | grep -E '^(EPS_|S3_|COMM_|PSI_|ONLY_|G2S3)' | sort > "$WORK/py.txt"; then
  sounio_foreign_diff "$WORK/souc.txt" "$WORK/py.txt" "g2s3" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "g2s3 gate: FAIL"; exit 1; }
echo "g2s3 gate: PASS"
