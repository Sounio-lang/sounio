#!/usr/bin/env bash
# scripts/stats_ols_diag_e2e_gate.sh — OLS diagnostics E2E under lean_single
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/stdlib/stats/test_ols_diag_e2e.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/ols_diag.elf"
LOG="$OUT/run.log"
fail=0

echo "== stats_ols_diag_e2e_gate: engine=$SOUNIO_SOUC_ENGINE =="
if ! "$SOUC" compile "$SRC" -o "$ELF" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: compile"; tail -40 "$OUT/compile.log" || true; fail=1
else
  chmod +x "$ELF"
  if ! "$ELF" >"$LOG" 2>&1; then
    echo "FAIL: run"; cat "$LOG" || true; fail=1
  elif ! grep -q "STATS_OLS_DIAG_E2E_OK" "$LOG"; then
    echo "FAIL: missing sentinel"; cat "$LOG" || true; fail=1
  else
    grep '^OLS_DIAG_E2E' "$LOG" || true
  fi
fi

if [[ $fail -eq 0 ]]; then
  # hard-check printed textbook slope/intercept tokens via oracle
  if ! grep -q "cook_slope=2.000000" "$LOG" && ! grep -q "cook_slope=2" "$LOG"; then
    # accept float print of 2.0
    if ! grep -E 'cook_slope=2(\.0+)?(\s|$)' "$LOG"; then
      echo "FAIL: cook slope not 2"; fail=1
    fi
  fi
  if ! grep -E 'max_cooks=2\.25' "$LOG"; then
    echo "FAIL: max cooks"; fail=1
  fi
  if ! grep -E 'ols_slope=0\.6' "$LOG"; then
    echo "FAIL: ols slope"; fail=1
  fi
fi

if [[ $fail -eq 0 ]]; then
  echo "STATS_OLS_DIAG_E2E_GATE_OK"
  exit 0
fi
exit 1
