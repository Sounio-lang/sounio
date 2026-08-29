#!/usr/bin/env bash
# Cross-toolchain replication gate for the 7-fiber structure of the sedenion ZD graph (Frente B, brick 2).
#
# WHY: souc v0.80.0 false-greens (silent stubs/miscompiles); a bare PASS is not proof of execution.
# This gate checks the 7 SPECIFIC fiber records (L, size, edges) emitted by souc are identical to the
# independent Python oracle, plus the participation/degree/intra-fiber invariants.
#
# Claim (tests/run-pass/sedenion_zd_fibers.sio): the 84 participating mixed-half primitives split into
# exactly 7 fibers indexed by L = lo XOR hi in {9..15}, each 12 vertices / 24 edges / degree 4, and
# annihilation never crosses fibers. (Companion facts "connected + bipartite 6+6" are oracle-verified.)
#
# Producers:
#   (1) souc   -> tests/run-pass/sedenion_zd_fibers.sio          emits `FIBER <code>` (L*10000+size*100+edges)
#   (2) python -> scripts/research/sedenion_zd_fibers_oracle.py  (transcribes ir_cd_sigma)
# Asserter: /usr/bin/diff. Exit 0 + CROSS-VERIFIED iff the 7 fiber records are identical AND both report
# PARTICIPATE 84 / DEGREE_BAD 0 / INTRA_BAD 0 / FIBERS OK (and the oracle: BIPARTITE_OK 7 / CONNECTED_OK 7).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() {
  if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
    "$SOUC" "$1" "$WORK/f.elf" >/dev/null 2>&1; chmod +x "$WORK/f.elf"; "$WORK/f.elf" 2>/dev/null; fi
}

echo "[zd-fibers] running souc + oracle ..."
run_souc tests/run-pass/sedenion_zd_fibers.sio > "$WORK/souc.txt"
python3 scripts/research/sedenion_zd_fibers_oracle.py > "$WORK/py.txt"

grep '^FIBER ' "$WORK/souc.txt" | awk '{print $2}' | sort -n > "$WORK/souc_fib.txt"
grep '^FIBER ' "$WORK/py.txt"   | awk '{print $2}' | sort -n > "$WORK/py_fib.txt"

field() { grep -m1 "^$1 " "$2" | awk '{print $2}'; }
fail=0
for key in PARTICIPATE DEGREE_BAD INTRA_BAD FIBERS; do
  s=$(field "$key" "$WORK/souc.txt"); p=$(field "$key" "$WORK/py.txt")
  if [ "$s" != "$p" ]; then echo "MISMATCH $key: souc=$s oracle=$p"; fail=1; fi
done
[ "$(field PARTICIPATE "$WORK/souc.txt")" = "84" ] || { echo "souc PARTICIPATE != 84"; fail=1; }
[ "$(field DEGREE_BAD "$WORK/souc.txt")" = "0" ]   || { echo "souc DEGREE_BAD != 0"; fail=1; }
[ "$(field INTRA_BAD "$WORK/souc.txt")" = "0" ]    || { echo "souc INTRA_BAD != 0"; fail=1; }
[ "$(field FIBERS "$WORK/souc.txt")" = "OK" ]      || { echo "souc FIBERS != OK"; fail=1; }
[ "$(field BIPARTITE_OK "$WORK/py.txt")" = "7" ]   || { echo "oracle BIPARTITE_OK != 7"; fail=1; }
[ "$(field CONNECTED_OK "$WORK/py.txt")" = "7" ]   || { echo "oracle CONNECTED_OK != 7"; fail=1; }

if [ "$(wc -l < "$WORK/souc_fib.txt")" -ne 7 ] || ! diff -q "$WORK/souc_fib.txt" "$WORK/py_fib.txt" >/dev/null; then
  echo "MISMATCH fiber records:"; diff "$WORK/souc_fib.txt" "$WORK/py_fib.txt" | head
  fail=1
fi

if [ "$fail" -ne 0 ]; then echo "zd-fibers gate: FAIL"; exit 1; fi
echo "CROSS-VERIFIED: 7/7 fiber records ELEMENT-WISE IDENTICAL (souc == Python oracle)"
echo "  84 vertices, 7 fibers (L=lo^hi in 9..15) x 12 vertices / 24 edges / degree 4, all edges intra-fiber."
echo "  oracle companions: every fiber connected + bipartite (6,6)."
echo "zd-fibers gate: PASS"
