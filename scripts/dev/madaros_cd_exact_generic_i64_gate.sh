#!/usr/bin/env bash
# scripts/dev/madaros_cd_exact_generic_i64_gate.sh
#
# Honest Madaros gate for tests/run-pass/cd_exact_generic_i64.sio under the
# default engine (no lean_single pin). This is the public residual that tip-green
# waves still list under claims_not_made.
#
# GREEN requires compile+run with the science sentinels:
#   ZD PROVED, SQ PASS, NONZERO PASS, and 16x "COMP <i> 0"
#
# Exit 0 + MADAROS_CD_EXACT_GENERIC_I64_GATE_OK only when green.
# RED is intentional and machine-readable — do not invent green.
#
# Usage:
#   bash scripts/dev/madaros_cd_exact_generic_i64_gate.sh
#   SOUC=./bin/souc bash scripts/dev/madaros_cd_exact_generic_i64_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
# Multi-module generic CD lowers can be stack- and memory-hungry.
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
TEST="tests/run-pass/cd_exact_generic_i64.sio"
OUT="$(mktemp -d /tmp/madaros_cd_exact_generic_i64.XXXXXX)"
trap 'rm -rf "$OUT"' EXIT

echo "== madaros_cd_exact_generic_i64_gate =="
echo "souc=$SOUC"
echo "test=$TEST"
if [[ ! -x "$SOUC" ]]; then
  echo "RED reason=souc_missing path=$SOUC" >&2
  echo "MADAROS_CD_EXACT_GENERIC_I64_GATE_FAIL" >&2
  exit 2
fi
if [[ ! -f "$ROOT/$TEST" ]]; then
  echo "RED reason=test_missing path=$TEST" >&2
  echo "MADAROS_CD_EXACT_GENERIC_I64_GATE_FAIL" >&2
  exit 2
fi

ENGINE_LINE="$("$SOUC" --version 2>&1 | head -1 || echo unknown)"
echo "engine=$ENGINE_LINE"

set +e
"$SOUC" compile "$ROOT/$TEST" -o "$OUT/cd.elf" >"$OUT/compile.log" 2>&1
compile_rc=$?
set -e

if [[ $compile_rc -ne 0 || ! -f "$OUT/cd.elf" || ! -s "$OUT/cd.elf" ]]; then
  echo "RED status=compile_fail rc=$compile_rc"
  echo "----- compile tail -----"
  tail -40 "$OUT/compile.log" || true
  # Compact error census for receipts / human triage.
  if command -v rg >/dev/null 2>&1; then
    echo "----- error census -----"
    rg -o 'error\[E[0-9]+' "$OUT/compile.log" 2>/dev/null | sort | uniq -c | sort -rn | head -20 || true
  fi
  echo "MADAROS_CD_EXACT_GENERIC_I64_GATE_FAIL" >&2
  exit 1
fi
chmod +x "$OUT/cd.elf" 2>/dev/null || true

set +e
"$OUT/cd.elf" >"$OUT/run.out" 2>"$OUT/run.err"
run_rc=$?
set -e

echo "----- run stdout -----"
cat "$OUT/run.out" || true
if [[ -s "$OUT/run.err" ]]; then
  echo "----- run stderr -----"
  cat "$OUT/run.err" || true
fi

if [[ $run_rc -ne 0 ]]; then
  echo "RED status=run_fail rc=$run_rc"
  echo "MADAROS_CD_EXACT_GENERIC_I64_GATE_FAIL" >&2
  exit 1
fi

fail=0
for tok in "ZD PROVED" "SQ PASS" "NONZERO PASS"; do
  if ! grep -qF "$tok" "$OUT/run.out"; then
    echo "RED missing_token=$tok"
    fail=1
  else
    echo "PASS token=$tok"
  fi
done

# Expect sixteen component lines proving the zero vector: COMP <i> 0
comp_zero=$(grep -cE '^COMP [0-9]+ 0$' "$OUT/run.out" || true)
if [[ "$comp_zero" -lt 16 ]]; then
  echo "RED status=comp_zero_count count=$comp_zero want>=16"
  fail=1
else
  echo "PASS COMP_zero_lines=$comp_zero"
fi

if [[ $fail -ne 0 ]]; then
  echo "MADAROS_CD_EXACT_GENERIC_I64_GATE_FAIL" >&2
  exit 1
fi

echo "GREEN cd_exact_generic_i64"
echo "MADAROS_CD_EXACT_GENERIC_I64_GATE_OK"
exit 0
