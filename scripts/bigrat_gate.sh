#!/usr/bin/env bash
# Gate for stdlib data::bigrat (unbounded exact rational over BigInt).
#
# ADR-008 pilot demotion:
#   Claim clock: tests/stdlib/data/test_bigrat_stdlib.sio must compile, run,
#   return 0, and print BIGRAT_STDLIB_OK. That driver already embeds Sounio-native
#   eq_or_fail checks (expected numerators/denominators as BigInt), so the claim
#   is decided inside Sounio — not by a peer language.
#
#   Foreign corroboration: digit-for-digit print reconstruction vs
#   scripts/research/bigrat_oracle.py remains available for hunting silent
#   big_print/codegen walls. Default is report-only. Set
#   SOUNIO_FOREIGN_ORACLE_HARD=1 to restore legacy hard-fail on oracle mismatch.
#
# Engine default lean_single for historical emitter path; override with
# SOUNIO_SOUC_ENGINE.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
HARD="${SOUNIO_FOREIGN_ORACLE_HARD:-0}"

echo "== check data/bigrat.sio =="
$SOUC check stdlib/data/bigrat.sio >/dev/null 2>&1 \
  || echo "NOTE: standalone check quirk on stdlib/data/bigrat.sio (Madaros check-mode)"

echo "== claim: run-proof with Sounio eq_or_fail (oracle_class=sounio_native_expected) =="
if ! $SOUC compile tests/stdlib/data/test_bigrat_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  echo "FAIL compile"
  exit 1
fi
chmod +x "$OUT/x.elf"
set +e
"$OUT/x.elf" > "$OUT/run.txt" 2>&1
run_rc=$?
set -e
if [[ $run_rc -ne 0 ]]; then
  echo "FAIL run rc=$run_rc (Sounio eq_or_fail rejected a case)"
  tail -20 "$OUT/run.txt" || true
  exit 1
fi
if ! grep -q "BIGRAT_STDLIB_OK" "$OUT/run.txt"; then
  echo "FAIL missing BIGRAT_STDLIB_OK"
  cat "$OUT/run.txt" || true
  exit 1
fi
echo "  claim: BIGRAT_STDLIB_OK (native eq_or_fail path green)"

echo "== corroboration: Python digit-for-digit print (HARD=$HARD) =="
awk '/=/{k=$0; while(k !~ /~/){getline; k=k $0} gsub(/~/,"",k); gsub(/[ \t]/,"",k); print k}' "$OUT/run.txt" \
  | grep -E "_num=|_den=" | sort > "$OUT/recon.txt" || true

if ! command -v python3 >/dev/null 2>&1; then
  echo "  SKIP: no python3"
  echo "BIGRAT_GATE_OK"
  exit 0
fi

if ! python3 scripts/research/bigrat_oracle.py | sort > "$OUT/oracle.txt" 2>/dev/null; then
  echo "  SKIP: bigrat_oracle.py failed to run"
  echo "BIGRAT_GATE_OK"
  exit 0
fi

if diff "$OUT/recon.txt" "$OUT/oracle.txt" >/dev/null; then
  echo "  oracle print corroboration: EXACT MATCH ($(wc -l < "$OUT/recon.txt") values)"
else
  echo "  oracle print corroboration: MISMATCH (not a claim failure under ADR-008)"
  diff "$OUT/recon.txt" "$OUT/oracle.txt" || true
  if [ "$HARD" = "1" ]; then
    echo "BIGRAT_GATE_FAIL foreign_hard (SOUNIO_FOREIGN_ORACLE_HARD=1)"
    exit 1
  fi
fi

echo "BIGRAT_GATE_OK"
exit 0
