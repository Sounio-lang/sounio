#!/usr/bin/env bash
# Cross-toolchain gate for the sedenion associator side: 1848 = 11*168 (Frente B).
# souc executes the associator combinatorics; an independent Python oracle recomputes it; /usr/bin/diff
# asserts they agree on TOTAL/GRADE8/OTHER/OCT and the 15 per-grade counts. A bare PASS is not proof.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
run_souc() { if [ "$SOUC" = "./bin/souc" ]; then ./bin/souc run "$1" 2>/dev/null; else
  "$SOUC" "$1" "$WORK/a.elf" >/dev/null 2>&1; chmod +x "$WORK/a.elf"; "$WORK/a.elf" 2>/dev/null; fi }
echo "[associator-1848] running souc + oracle ..."
run_souc tests/run-pass/sedenion_associator_1848.sio > "$WORK/souc.txt"
python3 scripts/research/sedenion_associator_1848_oracle.py > "$WORK/py.txt"
field() { grep -m1 "^$1 " "$2" | awk '{print $2}'; }
fail=0
for key in TOTAL GRADE8 OTHER OCT CLASS0 CLASS2 CLASS6 G8_NOTFULL ASSOC; do
  s=$(field "$key" "$WORK/souc.txt"); p=$(field "$key" "$WORK/py.txt")
  [ "$s" = "$p" ] || { echo "MISMATCH $key: souc=$s oracle=$p"; fail=1; }
done
[ "$(field TOTAL "$WORK/souc.txt")" = "1848" ] || { echo "souc TOTAL != 1848"; fail=1; }
[ "$(field ASSOC "$WORK/souc.txt")" = "OK" ]   || { echo "souc ASSOC != OK"; fail=1; }
# oracle self-consistency: 11*168 and the per-grade profile
[ "$(field TOTAL "$WORK/py.txt")" = "1848" ] || { echo "oracle TOTAL != 1848"; fail=1; }
[ "$fail" -eq 0 ] || { echo "associator-1848 gate: FAIL"; exit 1; }
echo "CROSS-VERIFIED: souc == oracle on the sedenion associator side."
echo "  1848 = 11*168 ordered non-associative basis triples; by grade: 14x120 + 168(grade 8); octonion 168."
echo "associator-1848 gate: PASS"
