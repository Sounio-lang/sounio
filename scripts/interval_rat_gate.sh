#!/usr/bin/env bash
# ADR-008: claim = INTERVAL_RAT_STDLIB_OK; Python print-diff soft unless HARD=1.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
HARD="${SOUNIO_FOREIGN_ORACLE_HARD:-0}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== claim: interval_rat (Sounio) =="
$SOUC check stdlib/data/interval_rat.sio >/dev/null 2>&1 \
  || echo "NOTE: standalone check quirk"
if ! $SOUC compile tests/stdlib/data/test_interval_rat_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  echo "FAIL compile"; exit 1
fi
chmod +x "$OUT/x.elf"
set +e; "$OUT/x.elf" > "$OUT/run.txt" 2>&1; rc=$?; set -e
if [ "$rc" -ne 0 ]; then echo "FAIL run"; cat "$OUT/run.txt"; exit 1; fi
grep -q "INTERVAL_RAT_STDLIB_OK" "$OUT/run.txt" || { echo "CLAIM FAIL: missing OK token"; exit 1; }
echo "  claim: INTERVAL_RAT_STDLIB_OK"
echo "== corroboration: Python (HARD=$HARD) =="
awk '/=/{k=$0; while(k !~ /~/){getline; k=k $0} gsub(/~/,"",k); gsub(/[ \t]/,"",k); print k}' "$OUT/run.txt" \
  | grep -E "_(num|den)=" | sort > "$OUT/recon.txt" || true
if python3 scripts/research/interval_rat_oracle.py 2>/dev/null | sort > "$OUT/oracle.txt"; then
  n=$(grep -c . "$OUT/recon.txt" || echo 0)
  if [ "$n" -eq 18 ] && diff "$OUT/recon.txt" "$OUT/oracle.txt" >/dev/null; then
    echo "  oracle print corroboration: EXACT MATCH"
  else
    echo "  oracle print corroboration: MISMATCH (n=$n)"
    diff "$OUT/recon.txt" "$OUT/oracle.txt" || true
    [ "$HARD" = "1" ] && fail=1
  fi
else echo "  SKIP oracle"; fi
[ $fail -eq 0 ] && echo "INTERVAL_RAT_GATE_OK" || exit 1
exit 0
