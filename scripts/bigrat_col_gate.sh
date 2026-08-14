#!/usr/bin/env bash
# ADR-008: claim = BIGRAT_COL_STDLIB_OK (+ compile/run); Python print-diff soft unless HARD=1.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
HARD="${SOUNIO_FOREIGN_ORACLE_HARD:-0}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== claim: bigrat column reduction (Sounio) =="
if ! $SOUC compile tests/stdlib/data/test_bigrat_col_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  echo "FAIL compile"; exit 1
fi
chmod +x "$OUT/x.elf"
set +e; "$OUT/x.elf" > "$OUT/run.txt" 2>&1; rc=$?; set -e
if [ "$rc" -ne 0 ]; then echo "FAIL run rc=$rc"; tail -20 "$OUT/run.txt"; exit 1; fi
grep -q "BIGRAT_COL_STDLIB_OK" "$OUT/run.txt" || { echo "CLAIM FAIL: missing OK token"; exit 1; }
echo "  claim: BIGRAT_COL_STDLIB_OK"
echo "== corroboration: Python print-diff (HARD=$HARD) =="
awk '/=/{k=$0; while(k !~ /~/){getline; k=k $0} gsub(/~/,"",k); gsub(/[ \t]/,"",k); print k}' "$OUT/run.txt" \
  | grep -E "col_.*(num|den)=" | sort > "$OUT/recon.txt" || true
if python3 scripts/research/bigrat_oracle.py col 2>/dev/null | sort > "$OUT/oracle.txt"; then
  n=$(grep -c . "$OUT/recon.txt" || echo 0)
  if [ "$n" -eq 6 ] && diff "$OUT/recon.txt" "$OUT/oracle.txt" >/dev/null; then
    echo "  oracle print corroboration: EXACT MATCH"
  else
    echo "  oracle print corroboration: MISMATCH (not claim under ADR-008)"
    diff "$OUT/recon.txt" "$OUT/oracle.txt" || true
    [ "$HARD" = "1" ] && fail=1
  fi
else
  echo "  SKIP: oracle failed"
fi
[ $fail -eq 0 ] && echo "BIGRAT_COL_GATE_OK" || { echo "BIGRAT_COL_GATE_FAIL"; exit 1; }
exit 0
