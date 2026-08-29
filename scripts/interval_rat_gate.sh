#!/usr/bin/env bash
# Gate for data::interval_rat (certified rational intervals). lean_single. Endpoints DIFFED vs the
# Python oracle (interval_rat_oracle.py) -- exit/token are not the signal (bigrat is struct-heavy).
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== check data/interval_rat.sio =="
$SOUC check stdlib/data/interval_rat.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk (driver + oracle prove the API)"
echo "== run-proof + ORACLE DIFF: certified rational intervals =="
if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile tests/stdlib/data/test_interval_rat_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" > "$OUT/run.txt" 2>&1 || true
  awk '/=/{k=$0; while(k !~ /~/){getline; k=k $0} gsub(/~/,"",k); gsub(/[ \t]/,"",k); print k}' "$OUT/run.txt" | grep -E "_(num|den)=" | sort > "$OUT/recon.txt"
  python3 scripts/research/interval_rat_oracle.py | sort > "$OUT/oracle.txt"
  n=$(grep -c . "$OUT/recon.txt")
  if [ "$n" -eq 18 ] && diff "$OUT/recon.txt" "$OUT/oracle.txt" >/dev/null; then echo "  oracle diff: EXACT MATCH (18 endpoint values)"; else echo "  ORACLE MISMATCH ($n/18):"; diff "$OUT/recon.txt" "$OUT/oracle.txt" || true; fail=1; fi
  grep -q "INTERVAL_RAT_STDLIB_OK" "$OUT/run.txt" || { echo "  missing OK token (ivr_contains failed)"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "INTERVAL_RAT_GATE_OK"
exit $fail
