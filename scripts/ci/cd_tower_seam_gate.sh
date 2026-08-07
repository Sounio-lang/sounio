#!/usr/bin/env bash
# ADR-008: claim = TOWER OK on Sounio; Python/ZDEQ64 soft unless HARD=1.
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/c.elf" >/dev/null 2>&1; chmod +x "$WORK/c.elf"; "$WORK/c.elf" 2>/dev/null; fi }
OUT="$(run_souc tests/run-pass/cd_tower_seam.sio)"
echo "$OUT" | grep -aE '^(LSQ|FWD|SEAM|ZDEQ|NA|OS)[0-9]' | sort > "$WORK/souc.txt" || true
fail=0
echo "$OUT" | grep -qa '^TOWER OK' || { echo "CLAIM FAIL: verdict != TOWER OK"; fail=1; }
if python3 scripts/research/cd_tower_seam_oracle.py > "$WORK/py_all.txt" 2>/dev/null; then
  # compare lines souc emitted
  grep -aE '^(LSQ|FWD|SEAM|ZDEQ|NA|OS)[0-9]' "$WORK/py_all.txt" | sort > "$WORK/py.txt" || true
  # original compared filtered set - use intersection by grepping common keys from souc
  sounio_foreign_diff "$WORK/souc.txt" "$WORK/py.txt" "cd-tower" || fail=1
  [ "$(grep '^ZDEQ64 ' "$WORK/py_all.txt" | awk '{print $2}')" = "1" ] \
    || sounio_foreign_mismatch "oracle ZDEQ64 != 1" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "cd-tower gate: FAIL"; exit 1; }
echo "cd-tower gate: PASS"
