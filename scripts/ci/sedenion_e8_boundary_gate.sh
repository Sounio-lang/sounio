#!/usr/bin/env bash
# ADR-008: claim = Sounio PARTICIPATE 84 / EXCLUDED 28 / INVARIANT HOLDS + 28 EXCL codes;
# Python set-identity is corroboration soft unless HARD=1.
set -euo pipefail
ROOT_FOR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_FOR_LIB/scripts/ci/lib_sounio_claim_oracle.sh"
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() {
  if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
    "$SOUC" "$1" "$WORK/e8.elf" >/dev/null 2>&1; chmod +x "$WORK/e8.elf"; "$WORK/e8.elf" 2>/dev/null; fi
}
echo "[e8-boundary] running souc + oracle ..."
run_souc tests/run-pass/sedenion_e8_boundary.sio > "$WORK/souc.txt"
python3 scripts/research/sedenion_e8_boundary_oracle.py > "$WORK/py.txt" || true
grep '^EXCL ' "$WORK/souc.txt" | awk '{print $2}' | sort -n > "$WORK/souc_excl.txt" || true
grep '^EXCL ' "$WORK/py.txt" 2>/dev/null | awk '{print $2}' | sort -n > "$WORK/py_excl.txt" || true
SN=$(wc -l < "$WORK/souc_excl.txt")
field() { grep -m1 "^$1 " "$2" | awk '{print $2}'; }
fail=0
[ "$(field PARTICIPATE "$WORK/souc.txt")" = "84" ] || { echo "CLAIM FAIL: souc PARTICIPATE != 84"; fail=1; }
[ "$(field EXCLUDED "$WORK/souc.txt")" = "28" ] || { echo "CLAIM FAIL: souc EXCLUDED != 28"; fail=1; }
[ "$(field INVARIANT "$WORK/souc.txt")" = "HOLDS" ] || { echo "CLAIM FAIL: souc INVARIANT != HOLDS"; fail=1; }
[ "$SN" -eq 28 ] || { echo "CLAIM FAIL: souc EXCL count $SN != 28"; fail=1; }
for key in PARTICIPATE EXCLUDED TOUCH_E8 DIAGONAL INVARIANT; do
  s=$(field "$key" "$WORK/souc.txt"); p=$(field "$key" "$WORK/py.txt" 2>/dev/null || true)
  if [ -n "${p:-}" ] && [ "$s" != "$p" ]; then
    sounio_foreign_mismatch "MISMATCH $key: souc=$s oracle=$p" || fail=1
  fi
done
if [ -s "$WORK/py_excl.txt" ]; then
  sounio_foreign_diff "$WORK/souc_excl.txt" "$WORK/py_excl.txt" "excluded set" || fail=1
fi
[ "$fail" -eq 0 ] || { echo "e8-boundary gate: FAIL"; exit 1; }
echo "e8-boundary gate: PASS"
