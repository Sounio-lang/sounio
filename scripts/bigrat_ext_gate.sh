#!/usr/bin/env bash
# Gate for data::bigrat extensions (sub/div/cmp/from_decimal/col_mean). lean_single. The big values are
# DIFFED against the Python oracle (bigrat_oracle.py ext) -- exit/token are not the signal (souc wall).
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== run-proof + ORACLE DIFF: bigrat extensions =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/data/test_bigrat_ext_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" > "$OUT/run.txt" 2>&1 || true
  awk '/=/{k=$0; while(k !~ /~/){getline; k=k $0} gsub(/~/,"",k); gsub(/[ \t]/,"",k); print k}' "$OUT/run.txt" | grep -E "(bigdec|colmean)_(num|den)=" | sort > "$OUT/recon.txt"
  python3 scripts/research/bigrat_oracle.py ext | sort > "$OUT/oracle.txt"
  n=$(grep -c . "$OUT/recon.txt")
  if [ "$n" -eq 4 ] && diff "$OUT/recon.txt" "$OUT/oracle.txt" >/dev/null; then echo "  oracle diff: EXACT MATCH (4 values)"; else echo "  ORACLE MISMATCH ($n/4):"; diff "$OUT/recon.txt" "$OUT/oracle.txt" || true; fail=1; fi
  grep -q "BIGRAT_EXT_STDLIB_OK" "$OUT/run.txt" || { echo "  missing OK token (a big_eq assertion failed)"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "BIGRAT_EXT_GATE_OK"
exit $fail
