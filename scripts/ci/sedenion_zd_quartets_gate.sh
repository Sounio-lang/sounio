#!/usr/bin/env bash
# ADR-008: claim = Sounio PAIRS/QUARTETS/QUARTETS OK; Python QMASK corroboration soft unless HARD=1.
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/q.elf" >/dev/null 2>&1; chmod +x "$WORK/q.elf"; "$WORK/q.elf" 2>/dev/null; fi }
echo "[zd-quartets] running souc + oracle ..."
run_souc tests/run-pass/sedenion_zd_quartets.sio > "$WORK/souc.txt"
python3 scripts/research/sedenion_zd_quartets_oracle.py > "$WORK/py.txt" || true
field() { grep -m1 "^$1 " "$2" | awk '{print $2}'; }
fail=0
[ "$(field PAIRS "$WORK/souc.txt")" = "168" ] || { echo "CLAIM FAIL: souc PAIRS != 168"; fail=1; }
[ "$(field QUARTETS "$WORK/souc.txt")" = "42" ] || { echo "CLAIM FAIL: souc QUARTETS != 42"; fail=1; }
[ "$(grep -c '^QUARTETS OK' "$WORK/souc.txt")" = "1" ] || { echo "CLAIM FAIL: missing QUARTETS OK"; fail=1; }
for key in PAIRS QUARTETS BAD_SIZE BAD_COUNT; do
  s=$(field "$key" "$WORK/souc.txt"); p=$(field "$key" "$WORK/py.txt" 2>/dev/null || true)
  if [ -n "${p:-}" ] && [ "$s" != "$p" ]; then
    sounio_foreign_mismatch "MISMATCH $key: souc=$s oracle=$p" || fail=1
  fi
done
[ "$(grep -c '^QMASK ' "$WORK/py.txt" 2>/dev/null || echo 0)" = "42" ] \
  || sounio_foreign_mismatch "oracle QMASK count != 42" || fail=1
[ "$(field QUARTETS_V "$WORK/py.txt" 2>/dev/null || true)" = "OK" ] \
  || sounio_foreign_mismatch "oracle verdict != OK" || fail=1
[ "$fail" -eq 0 ] || { echo "zd-quartets gate: FAIL"; exit 1; }
echo "zd-quartets gate: PASS"
