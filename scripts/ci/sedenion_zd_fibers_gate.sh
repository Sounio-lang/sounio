#!/usr/bin/env bash
# ADR-008: claim = Sounio PARTICIPATE/DEGREE/INTRA/FIBERS sentinels + 7 FIBER codes;
# Python BIPARTITE/CONNECTED and set-diff are corroboration (HARD=1 to fail).
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() {
  if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
    "$SOUC" "$1" "$WORK/f.elf" >/dev/null 2>&1; chmod +x "$WORK/f.elf"; "$WORK/f.elf" 2>/dev/null; fi
}
echo "[zd-fibers] running souc + oracle ..."
run_souc tests/run-pass/sedenion_zd_fibers.sio > "$WORK/souc.txt"
python3 scripts/research/sedenion_zd_fibers_oracle.py > "$WORK/py.txt" || true
grep '^FIBER ' "$WORK/souc.txt" | awk '{print $2}' | sort -n > "$WORK/souc_fib.txt" || true
grep '^FIBER ' "$WORK/py.txt" 2>/dev/null | awk '{print $2}' | sort -n > "$WORK/py_fib.txt" || true
field() { grep -m1 "^$1 " "$2" | awk '{print $2}'; }
fail=0
# claim
[ "$(field PARTICIPATE "$WORK/souc.txt")" = "84" ] || { echo "CLAIM FAIL: souc PARTICIPATE != 84"; fail=1; }
[ "$(field DEGREE_BAD "$WORK/souc.txt")" = "0" ]   || { echo "CLAIM FAIL: souc DEGREE_BAD != 0"; fail=1; }
[ "$(field INTRA_BAD "$WORK/souc.txt")" = "0" ]    || { echo "CLAIM FAIL: souc INTRA_BAD != 0"; fail=1; }
[ "$(field FIBERS "$WORK/souc.txt")" = "OK" ]      || { echo "CLAIM FAIL: souc FIBERS != OK"; fail=1; }
[ "$(wc -l < "$WORK/souc_fib.txt")" -eq 7 ]        || { echo "CLAIM FAIL: souc fiber count != 7"; fail=1; }
# foreign
for key in PARTICIPATE DEGREE_BAD INTRA_BAD FIBERS; do
  s=$(field "$key" "$WORK/souc.txt"); p=$(field "$key" "$WORK/py.txt" 2>/dev/null || true)
  if [ -n "${p:-}" ] && [ "$s" != "$p" ]; then
    sounio_foreign_mismatch "MISMATCH $key: souc=$s oracle=$p" || fail=1
  fi
done
[ "$(field BIPARTITE_OK "$WORK/py.txt" 2>/dev/null || true)" = "7" ] \
  || sounio_foreign_mismatch "oracle BIPARTITE_OK != 7" || fail=1
[ "$(field CONNECTED_OK "$WORK/py.txt" 2>/dev/null || true)" = "7" ] \
  || sounio_foreign_mismatch "oracle CONNECTED_OK != 7" || fail=1
if [ -s "$WORK/py_fib.txt" ]; then
  sounio_foreign_diff "$WORK/souc_fib.txt" "$WORK/py_fib.txt" "fiber records" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "zd-fibers gate: FAIL"; exit 1; }
echo "zd-fibers gate: PASS"
