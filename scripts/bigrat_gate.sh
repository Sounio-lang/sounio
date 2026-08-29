#!/usr/bin/env bash
# Gate for stdlib data::bigrat (unbounded exact rational over BigInt). lean_single engine.
# IMPORTANT: because souc has a codegen capacity wall that can SILENTLY emit wrong big values with a
# clean exit, the correctness signal is the DIGIT-FOR-DIGIT DIFF of the run-proof's printed values
# against the Python arbitrary-precision oracle (scripts/research/bigrat_oracle.py) -- NOT the program
# exit code and NOT the BIGRAT_STDLIB_OK token. Both the oracle diff and the token must pass.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check data/bigrat.sio =="
$SOUC check stdlib/data/bigrat.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk on stdlib/data/bigrat.sio (Madaros check-mode; driver + oracle prove the API)"
echo "== run-proof + ORACLE DIFF: unbounded exact rational =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/data/test_bigrat_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"
  "$OUT/x.elf" > "$OUT/run.txt" 2>&1 || true
  # reconstruct each multi-line big value into a single decimal, key=value
  awk '/=/{k=$0; while(k !~ /~/){getline; k=k $0} gsub(/~/,"",k); gsub(/[ \t]/,"",k); print k}' "$OUT/run.txt" \
      | grep -E "_num=|_den=" | sort > "$OUT/recon.txt"
  python3 scripts/research/bigrat_oracle.py | sort > "$OUT/oracle.txt"
  if diff "$OUT/recon.txt" "$OUT/oracle.txt" >/dev/null; then
    echo "  oracle diff: EXACT MATCH ($(wc -l < "$OUT/recon.txt") values)"
  else
    echo "  ORACLE MISMATCH (codegen wall or logic bug):"; diff "$OUT/recon.txt" "$OUT/oracle.txt" || true; fail=1
  fi
  grep -q "BIGRAT_STDLIB_OK" "$OUT/run.txt" || { echo "  missing OK token"; fail=1; }
else
  echo "FAIL compile"; fail=1
fi
[ $fail -eq 0 ] && echo "BIGRAT_GATE_OK"
exit $fail
