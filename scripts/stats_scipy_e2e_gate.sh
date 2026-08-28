#!/usr/bin/env bash
# scripts/stats_scipy_e2e_gate.sh
# scipy.stats-class epistemic E2E vertical under lean_single.
#
# Proves:
#   - Welch t + Levene + OLS + bootstrap + JB/QQ + paired t on fixed data
#   - TestResult.as_gum GUM bridge (structural surpass vs scipy.stats)
# Optional:
#   - pure-Python oracle (stdlib math; SciPy if installed) via stats_scipy_e2e_oracle.py
#
# Usage: bash scripts/stats_scipy_e2e_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUC:-$ROOT/bin/souc}"

SRC="tests/stdlib/stats/test_scipy_e2e_vertical.sio"
OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT
ELF="$OUT_DIR/scipy_e2e.elf"
LOG="$OUT_DIR/run.log"
RECEIPT_DIR="$ROOT/artifacts/stats"
RECEIPT="$RECEIPT_DIR/scipy_e2e_receipt.v1.json"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

fail=0

echo "== stats_scipy_e2e_gate: engine=$SOUNIO_SOUC_ENGINE =="
echo "== compile $SRC =="
if ! "$SOUC" compile "$SRC" -o "$ELF" >"$OUT_DIR/compile.log" 2>&1; then
  echo "FAIL: compile"
  tail -40 "$OUT_DIR/compile.log" || true
  fail=1
else
  chmod +x "$ELF"
  echo "== run =="
  if ! "$ELF" >"$LOG" 2>&1; then
    echo "FAIL: run exit non-zero"
    cat "$LOG" || true
    fail=1
  else
    if ! grep -q "STATS_SCIPY_E2E_OK" "$LOG"; then
      echo "FAIL: missing STATS_SCIPY_E2E_OK"
      cat "$LOG" || true
      fail=1
    else
      echo "---- metric lines ----"
      grep '^SCIPY_E2E' "$LOG" || true
      echo "----------------------"
    fi
  fi
fi

# Parse key metrics from stdout (best-effort)
parse_field() {
  # $1 = line prefix match, $2 = key name after SCIPY_E2E
  local line key
  line="$(grep -E "$1" "$LOG" 2>/dev/null | head -1 || true)"
  key="$2"
  if [[ -z "$line" ]]; then
    echo "null"
    return
  fi
  # extract key=value token
  echo "$line" | tr ' ' '\n' | grep -E "^${key}=" | head -1 | cut -d= -f2-
}

T_VAL="$(parse_field 'SCIPY_E2E t=' t)"
DF_VAL="$(parse_field 'SCIPY_E2E t=' df)"
P_VAL="$(parse_field 'SCIPY_E2E t=' p)"
D_VAL="$(parse_field 'SCIPY_E2E t=' d)"
GUM_U="$(parse_field 'SCIPY_E2E t=' gum_u)"
LEV_W="$(parse_field 'SCIPY_E2E t=' levene_w)"
OLS_S="$(parse_field 'SCIPY_E2E ols_slope=' ols_slope)"
OLS_I="$(parse_field 'SCIPY_E2E ols_slope=' ols_intercept)"
OLS_R2="$(parse_field 'SCIPY_E2E ols_slope=' ols_r2)"
BOOT_M="$(parse_field 'SCIPY_E2E boot_mean=' boot_mean)"
QQ_PPCC="$(parse_field 'SCIPY_E2E jb_p=' qq_ppcc)"
PAIRED_T="$(parse_field 'SCIPY_E2E paired_t=' paired_t)"

ORACLE_STATUS="skipped"
ORACLE_DETAIL="{}"
if [[ $fail -eq 0 ]] && [[ -f "$ROOT/scripts/stats_scipy_e2e_oracle.py" ]]; then
  echo "== optional python oracle =="
  set +e
  ORACLE_OUT="$(python3 "$ROOT/scripts/stats_scipy_e2e_oracle.py" \
    --t "$T_VAL" --df "$DF_VAL" --p "$P_VAL" --d "$D_VAL" \
    --levene-w "$LEV_W" \
    --ols-slope "$OLS_S" --ols-intercept "$OLS_I" --ols-r2 "$OLS_R2" \
    2>&1)"
  ORACLE_RC=$?
  set -e
  echo "$ORACLE_OUT"
  if [[ $ORACLE_RC -eq 0 ]]; then
    ORACLE_STATUS="pass"
    # last line may be JSON detail
    ORACLE_DETAIL="$(echo "$ORACLE_OUT" | tail -1)"
  elif [[ $ORACLE_RC -eq 2 ]]; then
    ORACLE_STATUS="skipped"
    ORACLE_DETAIL="$(echo "$ORACLE_OUT" | tail -1)"
  else
    ORACLE_STATUS="fail"
    ORACLE_DETAIL="$(echo "$ORACLE_OUT" | tail -1)"
    echo "FAIL: oracle mismatch"
    fail=1
  fi
fi

mkdir -p "$RECEIPT_DIR"
STATUS_STR="fail"
if [[ $fail -eq 0 ]]; then STATUS_STR="pass"; fi

# Minimal JSON receipt (no jq dependency)
cat >"$RECEIPT" <<EOF
{
  "schema": "stats.scipy_e2e_receipt.v1",
  "status": "$STATUS_STR",
  "engine": "$SOUNIO_SOUC_ENGINE",
  "commit": "$COMMIT",
  "source": "$SRC",
  "sounio": {
    "welch_t": $T_VAL,
    "welch_df": $DF_VAL,
    "welch_p": $P_VAL,
    "cohens_d": $D_VAL,
    "gum_u_c": $GUM_U,
    "levene_w": $LEV_W,
    "ols_slope": $OLS_S,
    "ols_intercept": $OLS_I,
    "ols_r2": $OLS_R2,
    "boot_mean": $BOOT_M,
    "qq_ppcc": $QQ_PPCC,
    "paired_t": $PAIRED_T
  },
  "epistemic": {
    "as_gum_present": true,
    "gum_u_c": $GUM_U,
    "note": "TestResult.as_gum bridges statistic ± SE via gum_simple; scipy.stats has no analogue"
  },
  "oracle": {
    "status": "$ORACLE_STATUS",
    "detail": $ORACLE_DETAIL
  },
  "claims_not_made": [
    "full_scipy_api",
    "shapiro_parity",
    "madaros_multimodule",
    "numpy_array_protocol"
  ]
}
EOF

echo "receipt: $RECEIPT"

if [[ $fail -eq 0 ]]; then
  echo "STATS_SCIPY_E2E_GATE_OK"
  exit 0
fi
exit 1
