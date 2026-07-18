#!/usr/bin/env bash
# Gate for data::bigrat_col_sum (exact rational column reduction over BigInt). lean_single.
# The correctness signal is the DIGIT-FOR-DIGIT diff of the printed decimals vs the Python oracle
# (scripts/research/bigrat_oracle.py col) -- NOT the exit code or OK token (souc can silently miscompute
# struct-heavy BigInt code). bigrat_col_sum is a loop (one add call-site) so it stays under the codegen
# capacity wall; the magnitude ceiling is LIMBS=64 (~10^576).
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== run-proof + ORACLE DIFF: bigrat column reduction =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/data/test_bigrat_col_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" > "$OUT/run.txt" 2>&1 || true
  awk '/=/{k=$0; while(k !~ /~/){getline; k=k $0} gsub(/~/,"",k); gsub(/[ \t]/,"",k); print k}' "$OUT/run.txt" | grep -E "col_.*(num|den)=" | sort > "$OUT/recon.txt"
  python3 scripts/research/bigrat_oracle.py col | sort > "$OUT/oracle.txt"
  n=$(grep -c . "$OUT/recon.txt")
  if [ "$n" -eq 6 ] && diff "$OUT/recon.txt" "$OUT/oracle.txt" >/dev/null; then
    echo "  oracle diff: EXACT MATCH (6 values; p100 denominator = $(awk -F= '/col_p100_den/{print length($2)}' "$OUT/oracle.txt") digits)"
  else
    echo "  ORACLE MISMATCH or missing output ($n/6 values):"; diff "$OUT/recon.txt" "$OUT/oracle.txt" || true; fail=1
  fi
  grep -q "BIGRAT_COL_STDLIB_OK" "$OUT/run.txt" || { echo "  missing OK token"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "BIGRAT_COL_GATE_OK"
exit $fail
